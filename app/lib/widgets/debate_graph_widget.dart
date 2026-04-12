import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:url_launcher/url_launcher.dart';

import '../l10n/app_strings.dart';

/// Interactive force-directed debate graph with pan/zoom, node/edge selection,
/// and an animated detail side-panel.
class DebateGraphWidget extends StatefulWidget {
  /// List of node maps: {id, name, team, role, color, specialty, statement_count}.
  final List<Map<String, dynamic>> nodes;

  /// List of edge maps: {id, source, target, relation_type, round}.
  final List<Map<String, dynamic>> edges;

  /// Optional custom team names for display.
  final String? teamAName;
  final String? teamBName;

  const DebateGraphWidget({
    super.key,
    required this.nodes,
    required this.edges,
    this.teamAName,
    this.teamBName,
  });

  @override
  State<DebateGraphWidget> createState() => _DebateGraphWidgetState();
}

class _DebateGraphWidgetState extends State<DebateGraphWidget>
    with SingleTickerProviderStateMixin {
  // -- layout state --
  final Map<String, Offset> _positions = {};
  final Map<String, Offset> _velocities = {};
  late Ticker _ticker;
  int _iteration = 0;
  static const int _maxIterations = 100;
  bool _layoutDone = false;
  bool _initialized = false;

  // -- interaction state --
  String? _selectedNodeId;
  String? _selectedEdgeId;
  String? _hoveredNodeId;
  bool _showEdgeLabels = false;
  Offset? _pointerDownPos;
  final TransformationController _transformCtrl = TransformationController();

  // -- panel animation --
  bool _panelVisible = false;

  // -- position snapshot (shared between rendering and hit testing) --
  Map<String, Offset> _posSnapshot = {};

  // -- parsed data --
  List<_GNode> _gNodes = [];
  List<_GEdge> _gEdges = [];

  // -- constants --
  static const double _nodeRadius = 14.0;
  static const double _panelWidth = 320.0;

  @override
  void initState() {
    super.initState();
    _ticker = createTicker(_onTick);
    _parseData();
  }

  @override
  void didUpdateWidget(covariant DebateGraphWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.nodes != widget.nodes || oldWidget.edges != widget.edges) {
      _parseData();
      // Preserve existing node positions, remove stale ones
      final currentIds = _gNodes.map((n) => n.id).toSet();
      _positions.removeWhere((k, _) => !currentIds.contains(k));
      _velocities.removeWhere((k, _) => !currentIds.contains(k));
      // Assign immediate positions to new nodes (don't wait for LayoutBuilder)
      final hasNewNodes = _gNodes.any((n) => !_positions.containsKey(n.id));
      if (hasNewNodes) {
        final rng = Random();
        for (final n in _gNodes) {
          if (!_positions.containsKey(n.id)) {
            _positions[n.id] = Offset(
              200 + rng.nextDouble() * 400,
              200 + rng.nextDouble() * 300,
            );
            _velocities[n.id] = Offset.zero;
          }
        }
        _iteration = 0;
        _layoutDone = false;
        if (!_ticker.isActive) _ticker.start();
      }
    }
  }

  @override
  void dispose() {
    _ticker.dispose();
    _transformCtrl.dispose();
    super.dispose();
  }

  // -- data parsing --

  void _parseData() {
    _gNodes = widget.nodes.asMap().entries.map((entry) {
      final i = entry.key;
      final m = entry.value;
      var rawId = m['id'] as String? ?? '';
      // Truncate very long IDs (e.g. JSON dict strings)
      if (rawId.length > 100) rawId = rawId.hashCode.toRadixString(16);
      return _GNode(
        id: rawId.isNotEmpty ? rawId : 'auto_node_$i',
        name: m['name'] as String? ?? m['label'] as String? ?? '',
        team: m['team'] as String? ?? '',
        role: m['role'] as String? ?? m['entity_type'] as String? ?? 'debater',
        color: m['color'] as String? ?? '',
        specialty: m['specialty'] as String? ?? '',
        statementCount: m['statement_count'] as int? ?? 0,
        currentRoundCount: m['current_round_count'] as int? ?? 0,
        uncitedRounds: m['uncited_rounds'] as int? ?? 0,
        citedInStatement: m['cited_in_statement'] as int? ?? 0,
        url: m['url'] as String? ?? '',
      );
    }).toList();

    _gEdges = widget.edges.map((m) {
      return _GEdge(
        id: m['id'] as String? ?? '',
        source: m['source'] as String? ?? '',
        target: m['target'] as String? ?? '',
        relationType: m['relation_type'] as String? ?? '',
        round: (m['round'] as int?) ?? (m['properties'] is Map ? (m['properties'] as Map)['round'] as int? ?? 0 : 0),
      );
    }).toList();
  }

  // -- initial random positions --

  void _initPositions(Size size) {
    if (_initialized) return;
    _initialized = true;

    // If there are no edges, use circular layout and skip force simulation.
    if (_gEdges.isEmpty && _gNodes.isNotEmpty) {
      final center = Offset(size.width / 2, size.height / 2);
      final radius = min(size.width, size.height) * 0.3;
      for (int i = 0; i < _gNodes.length; i++) {
        final angle = 2 * pi * i / _gNodes.length;
        _positions[_gNodes[i].id] = center + Offset(cos(angle) * radius, sin(angle) * radius);
      }
      _layoutDone = true;
      _normalizePositions();
      return;
    }

    final rng = Random(42);
    final cx = size.width / 2;
    final cy = size.height / 2;
    final spread = min(size.width, size.height) * 0.35;
    for (final n in _gNodes) {
      _positions[n.id] = Offset(
        cx + (rng.nextDouble() - 0.5) * spread * 2,
        cy + (rng.nextDouble() - 0.5) * spread * 2,
      );
      _velocities[n.id] = Offset.zero;
    }
    // Safety net: ensure ALL nodes have positions (catches edge cases)
    for (final n in _gNodes) {
      _positions.putIfAbsent(n.id, () => Offset(
        cx + (rng.nextDouble() - 0.5) * spread,
        cy + (rng.nextDouble() - 0.5) * spread,
      ));
      _velocities.putIfAbsent(n.id, () => Offset.zero);
    }
    if (!_ticker.isActive) _ticker.start();
  }

  /// Fruchterman-Reingold force-directed layout: repulsion between all nodes, attraction along edges.
  void _onTick(Duration _) {
    if (_layoutDone) {
      _ticker.stop();
      return;
    }
    _iteration++;
    if (_iteration > _maxIterations) {
      _layoutDone = true;
      _normalizePositions();
      _ticker.stop();
      setState(() {});
      return;
    }

    final area = 1200.0 * 1200.0;
    final k = sqrt(area / max(_gNodes.length, 1));
    final temperature = 10.0 * (1.0 - _iteration / _maxIterations);

    // Reset forces.
    final Map<String, Offset> forces = {};
    for (final n in _gNodes) {
      forces[n.id] = Offset.zero;
    }

    // Repulsion between all pairs (size-aware to prevent overlap).
    for (int i = 0; i < _gNodes.length; i++) {
      for (int j = i + 1; j < _gNodes.length; j++) {
        final a = _gNodes[i].id;
        final b = _gNodes[j].id;
        final pa = _positions[a];
        final pb = _positions[b];
        if (pa == null || pb == null) continue;
        var delta = pa - pb;
        var dist = delta.distance;
        if (dist < 0.01) {
          delta = const Offset(0.1, 0.1);
          dist = delta.distance;
        }
        // Minimum distance based on node radii
        final rA = _computeHitRadius(_gNodes[i]) - 10;  // remove buffer
        final rB = _computeHitRadius(_gNodes[j]) - 10;
        final minDist = rA + rB + 40;  // 40px gap
        final effectiveDist = max(dist, 1.0);
        var repForce = (k * k) / effectiveDist;
        // Extra push if nodes are overlapping
        if (dist < minDist) {
          repForce += (minDist - dist) * 5.0;
        }
        final normalized = delta / dist;
        forces[a] = forces[a]! + normalized * repForce;
        forces[b] = forces[b]! - normalized * repForce;
      }
    }

    // Attraction along edges.
    for (final e in _gEdges) {
      final pa = _posSnapshot[e.source];
      final pb = _posSnapshot[e.target];
      if (pa == null || pb == null) continue;
      var delta = pb - pa;
      var dist = delta.distance;
      if (dist < 0.01) continue;
      final attForce = (dist * dist) / k;
      final normalized = delta / dist;
      if (forces.containsKey(e.source)) forces[e.source] = forces[e.source]! + normalized * attForce;
      if (forces.containsKey(e.target)) forces[e.target] = forces[e.target]! - normalized * attForce;
    }

    // Centering force — pull all nodes gently toward graph center
    if (_gNodes.isNotEmpty) {
      double cx = 0, cy = 0;
      for (final n in _gNodes) {
        final p = _positions[n.id];
        if (p != null) { cx += p.dx; cy += p.dy; }
      }
      cx /= _gNodes.length;
      cy /= _gNodes.length;
      final graphCenter = Offset(cx, cy);
      for (final n in _gNodes) {
        final pos = _positions[n.id];
        if (pos == null) continue;
        final toCenter = graphCenter - pos;
        final centerDist = toCenter.distance;
        if (centerDist > 1.0) {
          forces[n.id] = forces[n.id]! + (toCenter / centerDist) * centerDist * 0.07;
        }
      }
    }

    // Apply forces with temperature limiting + track total movement.
    double totalMovement = 0;
    for (final n in _gNodes) {
      // (dragging removed — no node to skip)
      final f = forces[n.id];
      if (f == null) continue;
      final fLen = f.distance;
      if (fLen < 0.01) continue;
      final capped = f / fLen * min(fLen, temperature);
      final oldPos = _positions[n.id];
      if (oldPos == null) continue;
      _positions[n.id] = oldPos + capped;
      totalMovement += capped.distance;
    }

    // Post-process: resolve all overlaps by pushing nodes apart
    _resolveOverlaps();

    // Early convergence: stop if layout has settled
    if (totalMovement < 1.0 && _iteration > 10) {
      _layoutDone = true;
      _resolveOverlaps(); // final pass
      _normalizePositions();
      _ticker.stop();
    }

    if (_iteration % 2 == 0 || _layoutDone) {
      if (mounted) setState(() {});
    }
  }

  /// Push overlapping nodes apart until no overlap remains.
  void _resolveOverlaps() {
    for (int pass = 0; pass < 3; pass++) {
      for (int i = 0; i < _gNodes.length; i++) {
        for (int j = i + 1; j < _gNodes.length; j++) {
          final a = _gNodes[i].id, b = _gNodes[j].id;
          final pa = _positions[a], pb = _positions[b];
          if (pa == null || pb == null) continue;

          final rA = _computeHitRadius(_gNodes[i]) - 10; // visual radius
          final rB = _computeHitRadius(_gNodes[j]) - 10;
          final minDist = rA + rB + 20; // 20px minimum gap

          final delta = pb - pa;
          final dist = delta.distance;
          if (dist < minDist && dist > 0.01) {
            final overlap = (minDist - dist) / 2;
            final dir = delta / dist;
            _positions[a] = pa - dir * overlap;
            _positions[b] = pb + dir * overlap;
          } else if (dist <= 0.01) {
            // Identical positions — push apart randomly
            _positions[b] = pb + Offset(minDist * 0.5, minDist * 0.3);
          }
        }
      }
    }
  }

  /// Normalize all positions to positive coordinates (min 100px padding).
  /// Preserves relative layout, just shifts everything so nothing is negative.
  void _normalizePositions() {
    if (_gNodes.isEmpty) return;
    double minX = double.infinity, minY = double.infinity;
    for (final n in _gNodes) {
      final p = _positions[n.id];
      if (p != null) {
        if (p.dx < minX) minX = p.dx;
        if (p.dy < minY) minY = p.dy;
      }
    }
    if (minX < 100 || minY < 100) {
      final dx = minX < 100 ? 100 - minX : 0.0;
      final dy = minY < 100 ? 100 - minY : 0.0;
      for (final n in _gNodes) {
        final p = _positions[n.id];
        if (p != null) {
          _positions[n.id] = Offset(p.dx + dx, p.dy + dy);
        }
      }
    }
  }

  // -- hit testing --

  double _computeHitRadius(_GNode n) {
    final growthPerCite = (n.role == 'evidence' && n.citedInStatement == 0) ? 0.8 : 4.0;
    final fullRadius = _nodeRadius + (n.statementCount * growthPerCite).clamp(0.0, 40.0);
    final adjustedRadius = n.role == 'evidence' && n.uncitedRounds > 0
        ? (fullRadius - n.uncitedRounds * 4.0).clamp(4.0, fullRadius)
        : fullRadius;
    return adjustedRadius + 10;  // 10px click buffer
  }

  String? _hitTestNode(Offset pos, {bool logNearest = false}) {
    String? closestId;
    double closestDist = double.infinity;
    String? closestName;
    Offset? closestPos;
    double closestHitR = 0;
    for (final n in _gNodes) {
      final np = _posSnapshot[n.id];
      if (np == null) continue;
      final hitRadius = _computeHitRadius(n);
      final dist = (pos - np).distance;
      if (dist < closestDist) {
        closestDist = dist;
        closestId = dist <= hitRadius ? n.id : null;
        closestName = n.name;
        closestPos = np;
        closestHitR = hitRadius;
      }
      if (dist <= hitRadius && dist < closestDist) {
        closestId = n.id;
      }
    }
    if (logNearest) {
      debugPrint('[NEAREST] id=${closestId ?? "MISS"} name=$closestName pos=$closestPos dist=${closestDist.toStringAsFixed(1)} hitR=${closestHitR.toStringAsFixed(1)}');
    }
    return closestId;
  }

  String? _hitTestEdge(Offset pos) {
    const threshold = 8.0;
    // Build multi-edge index for curvature.
    final pairCounts = <String, int>{};
    final pairIndex = <String, int>{};
    for (final e in _gEdges) {
      final key = _pairKey(e.source, e.target);
      pairCounts[key] = (pairCounts[key] ?? 0) + 1;
    }
    final pairCurrent = <String, int>{};
    for (final e in _gEdges) {
      final key = _pairKey(e.source, e.target);
      pairCurrent[key] = (pairCurrent[key] ?? 0) + 1;
      pairIndex[e.id] = pairCurrent[key]!;
    }

    for (final e in _gEdges) {
      final pa = _posSnapshot[e.source];
      final pb = _posSnapshot[e.target];
      if (pa == null || pb == null) continue;

      final key = _pairKey(e.source, e.target);
      final count = pairCounts[key] ?? 1;
      final idx = pairIndex[e.id] ?? 1;

      if (count <= 1) {
        // Straight line hit test.
        final d = _distToSegment(pos, pa, pb);
        if (d < threshold) return e.id;
      } else {
        // Bezier hit test (sample points).
        final curvature = _curvatureForIndex(idx, count);
        final mid = (pa + pb) / 2;
        final dir = pb - pa;
        final perp = Offset(-dir.dy, dir.dx);
        final pLen = perp.distance;
        final ctrl =
            pLen > 0.01 ? mid + (perp / pLen) * curvature : mid;
        for (double t = 0; t <= 1.0; t += 0.05) {
          final pt = _quadBezier(pa, ctrl, pb, t);
          if ((pos - pt).distance < threshold) return e.id;
        }
      }
    }
    return null;
  }

  double _distToSegment(Offset p, Offset a, Offset b) {
    final ab = b - a;
    final ap = p - a;
    final lenSq = ab.dx * ab.dx + ab.dy * ab.dy;
    if (lenSq < 0.001) return (p - a).distance;
    var t = (ap.dx * ab.dx + ap.dy * ab.dy) / lenSq;
    t = t.clamp(0.0, 1.0);
    final proj = a + ab * t;
    return (p - proj).distance;
  }

  Offset _quadBezier(Offset a, Offset ctrl, Offset b, double t) {
    final u = 1.0 - t;
    return a * (u * u) + ctrl * (2 * u * t) + b * (t * t);
  }

  String _pairKey(String a, String b) {
    return a.compareTo(b) < 0 ? '$a|$b' : '$b|$a';
  }

  double _curvatureForIndex(int idx, int count) {
    if (count <= 1) return 0;
    // Distribute parallel edges symmetrically: +-40, +-80, +-120...
    final half = (idx + 1) ~/ 2;
    final sign = idx.isOdd ? 1.0 : -1.0;
    return sign * half * 40.0;
  }

  // -- color maps --

  static const Map<String, Color> _defaultEdgeColors = {
    'filed_against': Colors.orange,
    'represented_by': Colors.blue,
    'violates': Colors.red,
    'cites': Colors.blue,
    'references': Colors.blue,
    'ruled_by': Colors.amber,
    'related_to': Color(0xFF999999),
    'part_of': Colors.teal,
    'occurred_at': Colors.indigo,
    'contracted_with': Colors.green,
    'rebuts': Colors.red,
    'opposes': Colors.red,
    'supports': Colors.green,
    'agrees': Colors.green,
    'discusses_with': Colors.teal,
    'questions': Colors.deepOrange,
  };

  Color _nodeColor(_GNode node) {
    // Use hex color from backend if available
    if (node.color.isNotEmpty && node.color.startsWith('#') && node.color.length >= 7) {
      try {
        return Color(int.parse('FF${node.color.substring(1)}', radix: 16));
      } catch (_) {}
    }
    // Fallback by team/role
    if (node.team == 'team_a') return const Color(0xFF4A90D9);
    if (node.team == 'team_b') return const Color(0xFFE74C3C);
    if (node.role == 'judge') return Colors.amber;
    if (node.role == 'evidence') return const Color(0xFF9E9E9E);
    return Colors.blueGrey;
  }

  /// Returns legend items based on team/role, matching actual node colors.
  List<MapEntry<String, Color>> _buildLegendItems() {
    final items = <String, Color>{};
    for (final node in _gNodes) {
      if (node.team == 'team_a') items[S.teamName('team_a', teamAName: widget.teamAName, teamBName: widget.teamBName)] = const Color(0xFF4A90D9);
      else if (node.team == 'team_b') items[S.teamName('team_b', teamAName: widget.teamAName, teamBName: widget.teamBName)] = const Color(0xFFE74C3C);
      else if (node.role == 'judge') items[S.get('judges_label')] = Colors.amber;
      else if (node.role == 'evidence') items[S.get('evidence_label')] = const Color(0xFF9E9E9E);
    }
    return items.entries.toList();
  }

  Color _edgeColor(String relationType) {
    return _defaultEdgeColors[relationType.toLowerCase()] ?? const Color(0xFFCCCCCC);
  }

  // -- helpers --

  _GNode? _nodeById(String id) {
    for (final n in _gNodes) {
      if (n.id == id) return n;
    }
    return null;
  }

  _GEdge? _edgeById(String id) {
    for (final e in _gEdges) {
      if (e.id == id) return e;
    }
    return null;
  }

  List<_GEdge> _edgesForNode(String nodeId) {
    return _gEdges
        .where((e) => e.source == nodeId || e.target == nodeId)
        .toList();
  }

  // -- build --

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final size = Size(constraints.maxWidth, constraints.maxHeight);
        _initPositions(size);

        // Create snapshot for consistent rendering + hit testing
        _posSnapshot = Map.of(_positions);

        return Stack(
          children: [
            // Graph with zoom/pan
            InteractiveViewer(
              transformationController: _transformCtrl,
              minScale: 0.05,
              maxScale: 3.0,
              boundaryMargin: const EdgeInsets.all(double.infinity),
              child: CustomPaint(
                    size: const Size(10000, 10000),
                    painter: _GraphPainter(
                      nodes: _gNodes,
                      edges: _gEdges,
                      positions: _posSnapshot,
                      selectedNodeId: _selectedNodeId,
                      selectedEdgeId: _selectedEdgeId,
                      hoveredNodeId: _hoveredNodeId,
                      showEdgeLabels: _showEdgeLabels,
                      nodeColorFn: _nodeColor,
                      edgeColorFn: _edgeColor,
                      layoutDone: _layoutDone,
                    ),
                  ),
              ),

            // Click + hover handler (above InteractiveViewer, uses toScene)
            Positioned.fill(
              child: Listener(
                behavior: HitTestBehavior.translucent,
                onPointerDown: (e) {
                  _pointerDownPos = e.localPosition;
                },
                onPointerUp: (e) {
                  final downPos = _pointerDownPos;
                  _pointerDownPos = null;
                  if (downPos == null) return;
                  if ((e.localPosition - downPos).distance > 10) return;
                  final canvasPos = _transformCtrl.toScene(e.localPosition);
                  final nodeId = _hitTestNode(canvasPos);
                  if (nodeId != null) {
                    setState(() { _selectedNodeId = nodeId; _selectedEdgeId = null; _panelVisible = true; });
                    return;
                  }
                  final edgeId = _hitTestEdge(canvasPos);
                  if (edgeId != null) {
                    setState(() { _selectedEdgeId = edgeId; _selectedNodeId = null; _panelVisible = true; });
                    return;
                  }
                  setState(() { _selectedNodeId = null; _selectedEdgeId = null; _panelVisible = false; });
                },
                onPointerHover: (e) {
                  final canvasPos = _transformCtrl.toScene(e.localPosition);
                  String? hovId;
                  double bestDist = double.infinity;
                  for (final n in _gNodes) {
                    final np = _posSnapshot[n.id];
                    if (np == null) continue;
                    final d = (canvasPos - np).distance;
                    if (d <= _computeHitRadius(n) && d < bestDist) {
                      bestDist = d;
                      hovId = n.id;
                    }
                  }
                  if (hovId != _hoveredNodeId) {
                    setState(() => _hoveredNodeId = hovId);
                  }
                },
              ),
            ),

            // Toggle edge labels button
            Positioned(
              top: 8,
              left: 8,
              child: Material(
                color: Colors.white,
                elevation: 1,
                borderRadius: BorderRadius.circular(4),
                child: InkWell(
                  borderRadius: BorderRadius.circular(4),
                  onTap: () =>
                      setState(() => _showEdgeLabels = !_showEdgeLabels),
                  child: Padding(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          _showEdgeLabels
                              ? Icons.label
                              : Icons.label_off,
                          size: 16,
                          color: Colors.black54,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          _showEdgeLabels ? S.get('hide_labels') : S.get('show_labels'),
                          style: const TextStyle(
                            fontSize: 12,
                            color: Colors.black54,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),

            // Reset view button (top-left, next to edge labels)
            Positioned(
              top: 8,
              left: 140,
              child: Material(
                color: Colors.white,
                elevation: 1,
                borderRadius: BorderRadius.circular(4),
                child: InkWell(
                  borderRadius: BorderRadius.circular(4),
                  onTap: () => _transformCtrl.value = Matrix4.identity(),
                  child: Padding(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(
                          Icons.center_focus_strong,
                          size: 16,
                          color: Colors.black54,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          S.get('reset_view'),
                          style: const TextStyle(
                            fontSize: 12,
                            color: Colors.black54,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),

            // Legend (bottom-left) — IgnorePointer so nodes behind can be clicked
            Positioned(
              bottom: 8,
              left: 8,
              child: IgnorePointer(
                child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.white,
                  border: Border.all(color: const Color(0xFFE5E5E5)),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Wrap(
                  spacing: 12,
                  runSpacing: 4,
                  children: _buildLegendItems().map((entry) {
                    return Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: 8, height: 8,
                          decoration: BoxDecoration(
                            color: entry.value,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 4),
                        Text(
                          entry.key,
                          style: const TextStyle(fontSize: 10, color: Color(0xFF666666)),
                        ),
                      ],
                    );
                  }).toList(),
                ),
              ),
              ),  // IgnorePointer
            ),

            // Detail panel
            AnimatedPositioned(
              duration: const Duration(milliseconds: 300),
              curve: Curves.easeOut,
              top: 0,
              bottom: 0,
              right: _panelVisible ? 0 : -_panelWidth,
              width: _panelWidth,
              child: _buildDetailPanel(),
            ),
          ],
        );
      },
    );
  }

  // -- detail panel --

  Widget _buildDetailPanel() {
    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        border: Border(
          left: BorderSide(color: Color(0xFFEAEAEA)),
        ),
      ),
      child: _selectedNodeId != null
          ? _buildNodeDetail(_nodeById(_selectedNodeId!)!)
          : _selectedEdgeId != null
              ? _buildEdgeDetail(_edgeById(_selectedEdgeId!)!)
              : const SizedBox.shrink(),
    );
  }

  Widget _buildNodeDetail(_GNode node) {
    final connectedEdges = _edgesForNode(node.id);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header row
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 8, 8),
          child: Row(
            children: [
              Expanded(
                child: Row(
                  children: [
                    Flexible(
                      child: Text(
                        node.name,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    const SizedBox(width: 8),
                    _typeBadge(node.role, _nodeColor(node)),
                  ],
                ),
              ),
              IconButton(
                icon: const Icon(Icons.close, size: 18),
                onPressed: () => setState(() {
                  _selectedNodeId = null;
                  _panelVisible = false;
                }),
              ),
            ],
          ),
        ),
        const Divider(height: 1),
        // Details
        Expanded(
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              if (node.team.isNotEmpty) ...[
                _propRow('Team', S.teamName(node.team, teamAName: widget.teamAName, teamBName: widget.teamBName)),
              ],
              if (node.specialty.isNotEmpty) ...[
                _propRow('Specialty', node.specialty),
              ],
              if (node.statementCount > 0) ...[
                _propRow('Statements', '${node.statementCount}'),
              ],
              if (node.url.isNotEmpty) ...[
                Padding(
                  padding: const EdgeInsets.only(bottom: 4),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      SizedBox(
                        width: 90,
                        child: Text('Link', style: TextStyle(fontSize: 12, color: Color(0xFF999999))),
                      ),
                      Expanded(
                        child: InkWell(
                          onTap: () => launchUrl(Uri.parse(node.url), mode: LaunchMode.externalApplication),
                          child: Text(
                            'law.go.kr',
                            style: TextStyle(fontSize: 13, color: Colors.blue, decoration: TextDecoration.underline),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
              if (node.team.isNotEmpty || node.specialty.isNotEmpty || node.statementCount > 0 || node.url.isNotEmpty)
                const Divider(height: 24),
              const Text(
                'Connected Relations',
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: Colors.black87,
                ),
              ),
              const SizedBox(height: 8),
              if (connectedEdges.isEmpty)
                const Text(
                  'No relations',
                  style: TextStyle(fontSize: 12, color: Colors.black38),
                )
              else
                ...connectedEdges.map((e) {
                  final isSource = e.source == node.id;
                  final otherNode =
                      _nodeById(isSource ? e.target : e.source);
                  final otherLabel = otherNode?.name ?? '?';
                  return InkWell(
                    onTap: () => setState(() {
                      _selectedEdgeId = e.id;
                      _selectedNodeId = null;
                    }),
                    child: Padding(
                      padding: const EdgeInsets.symmetric(vertical: 4),
                      child: Row(
                        children: [
                          Icon(
                            isSource
                                ? Icons.arrow_forward
                                : Icons.arrow_back,
                            size: 14,
                            color: _edgeColor(e.relationType),
                          ),
                          const SizedBox(width: 6),
                          Text(
                            '[${e.relationType}]',
                            style: TextStyle(
                              fontSize: 12,
                              color: _edgeColor(e.relationType),
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                          const SizedBox(width: 6),
                          Expanded(
                            child: Text(
                              otherLabel,
                              style: const TextStyle(fontSize: 12),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                        ],
                      ),
                    ),
                  );
                }),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildEdgeDetail(_GEdge edge) {
    final srcNode = _nodeById(edge.source);
    final tgtNode = _nodeById(edge.target);
    final srcLabel = srcNode?.name ?? edge.source;
    final tgtLabel = tgtNode?.name ?? edge.target;
    final edgeColor = _edgeColor(edge.relationType);

    void _goToNode(String nodeId) {
      final pos = _posSnapshot[nodeId];
      final m = _transformCtrl.value;
      final tx = m.getTranslation().x.toStringAsFixed(0);
      final ty = m.getTranslation().y.toStringAsFixed(0);
      final sc = m.getMaxScaleOnAxis().toStringAsFixed(2);
      debugPrint('[EDGE→NODE] id=$nodeId, pos=$pos, translate=($tx,$ty), scale=$sc');
      setState(() {
        _selectedNodeId = nodeId;
        _selectedEdgeId = null;
      });
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header: relation_type + round + close button
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 8, 8),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: edgeColor.withAlpha(25),
                  border: Border.all(color: edgeColor.withAlpha(77)),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  edge.relationType,
                  style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: edgeColor),
                ),
              ),
              if (edge.round > 0) ...[
                const SizedBox(width: 8),
                Text('Round ${edge.round}', style: const TextStyle(fontSize: 12, color: Color(0xFF999999))),
              ],
              const Spacer(),
              IconButton(
                icon: const Icon(Icons.close, size: 18),
                onPressed: () => setState(() {
                  _selectedEdgeId = null;
                  _panelVisible = false;
                }),
              ),
            ],
          ),
        ),
        const Divider(height: 1),
        // Properties + Connected Nodes (scrollable, same layout as Node detail)
        Expanded(
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              // Properties
              const Text(
                'Properties',
                style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Colors.black87),
              ),
              const SizedBox(height: 8),
              _propRow('relation_type', edge.relationType),
              if (edge.round > 0)
                _propRow('round', '${edge.round}'),
              const SizedBox(height: 12),
              const Divider(height: 1),
              const SizedBox(height: 12),
              // Connected Nodes
              const Text(
                'Connected Nodes',
                style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Colors.black87),
              ),
              const SizedBox(height: 8),
              // Source node
              InkWell(
                onTap: () => _goToNode(edge.source),
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    children: [
                      Icon(Icons.arrow_back, size: 14, color: edgeColor),
                      const SizedBox(width: 6),
                      Text('[${edge.relationType}]',
                          style: TextStyle(fontSize: 12, color: edgeColor, fontWeight: FontWeight.w500)),
                      const SizedBox(width: 4),
                      Expanded(
                        child: Text(srcLabel,
                            style: const TextStyle(fontSize: 13, color: Colors.blue),
                            overflow: TextOverflow.ellipsis),
                      ),
                    ],
                  ),
                ),
              ),
              // Target node
              InkWell(
                onTap: () => _goToNode(edge.target),
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    children: [
                      Icon(Icons.arrow_forward, size: 14, color: edgeColor),
                      const SizedBox(width: 6),
                      Text('[${edge.relationType}]',
                          style: TextStyle(fontSize: 12, color: edgeColor, fontWeight: FontWeight.w500)),
                      const SizedBox(width: 4),
                      Expanded(
                        child: Text(tgtLabel,
                            style: const TextStyle(fontSize: 13, color: Colors.blue),
                            overflow: TextOverflow.ellipsis),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _propRow(String key, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 100,
            child: Text(
              key,
              style: const TextStyle(fontSize: 12, color: Colors.black54),
            ),
          ),
          Expanded(
            child: Text(value, style: const TextStyle(fontSize: 12)),
          ),
        ],
      ),
    );
  }

  Widget _typeBadge(String label, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color.withValues(alpha: 0.4)),
      ),
      child: Text(
        label,
        style: TextStyle(fontSize: 10, color: color, fontWeight: FontWeight.w600),
      ),
    );
  }
}

class _GNode {
  final String id;
  final String name;
  final String team;     // "team_a", "team_b", "" (judge/evidence)
  final String role;     // "debater", "judge", "evidence"
  final String color;    // hex color like "#4A90D9"
  final String specialty;
  final int statementCount;
  final int currentRoundCount;
  final int uncitedRounds;
  final int citedInStatement;
  final String url;

  const _GNode({
    required this.id,
    required this.name,
    this.team = '',
    this.role = 'debater',
    this.color = '',
    this.specialty = '',
    this.statementCount = 0,
    this.currentRoundCount = 0,
    this.uncitedRounds = 0,
    this.citedInStatement = 0,
    this.url = '',
  });
}

class _GEdge {
  final String id;
  final String source;
  final String target;
  final String relationType;  // "rebuts", "supports", "cites"
  final int round;

  const _GEdge({
    required this.id,
    required this.source,
    required this.target,
    this.relationType = '',
    this.round = 0,
  });
}

class _GraphPainter extends CustomPainter {
  final List<_GNode> nodes;
  final List<_GEdge> edges;
  final Map<String, Offset> positions;
  final String? selectedNodeId;
  final String? selectedEdgeId;
  final String? hoveredNodeId;
  final bool showEdgeLabels;
  final Color Function(_GNode) nodeColorFn;
  final Color Function(String) edgeColorFn;
  final bool layoutDone;

  static const double _nodeRadius = 14.0;

  final Map<String, TextPainter> _labelCache = {};

  _GraphPainter({
    required this.nodes,
    required this.edges,
    required this.positions,
    this.selectedNodeId,
    this.selectedEdgeId,
    this.hoveredNodeId,
    required this.showEdgeLabels,
    required this.nodeColorFn,
    required this.edgeColorFn,
    required this.layoutDone,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Background
    canvas.drawRect(
      Rect.fromLTWH(0, 0, size.width, size.height),
      Paint()..color = Colors.white,
    );

    // Build multi-edge counts
    final pairCounts = <String, int>{};
    for (final e in edges) {
      final key = _pairKey(e.source, e.target);
      pairCounts[key] = (pairCounts[key] ?? 0) + 1;
    }
    final pairCurrent = <String, int>{};

    // Draw edges
    for (final e in edges) {
      final key = _pairKey(e.source, e.target);
      pairCurrent[key] = (pairCurrent[key] ?? 0) + 1;
      final idx = pairCurrent[key]!;
      final count = pairCounts[key] ?? 1;
      _drawEdge(canvas, e, idx, count);
    }

    // Draw nodes
    for (final n in nodes) {
      _drawNode(canvas, n);
    }
  }

  void _drawEdge(Canvas canvas, _GEdge edge, int idx, int count) {
    final pa = positions[edge.source];
    final pb = positions[edge.target];
    if (pa == null || pb == null) return;

    final isSelected = edge.id == selectedEdgeId;
    final color = edgeColorFn(edge.relationType);
    final strokeWidth = isSelected ? 3.0 : 1.5;

    final paint = Paint()
      ..color = isSelected ? color : color.withValues(alpha: 0.7)
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke;

    if (count <= 1) {
      // Straight line
      canvas.drawLine(pa, pb, paint);

      if (showEdgeLabels) {
        _drawEdgeLabel(canvas, (pa + pb) / 2, edge.relationType);
      }
    } else {
      // Bezier curve
      final curvature = _curvatureForIndex(idx, count);
      final mid = (pa + pb) / 2;
      final dir = pb - pa;
      final perp = Offset(-dir.dy, dir.dx);
      final pLen = perp.distance;
      final ctrl = pLen > 0.01 ? mid + (perp / pLen) * curvature : mid;

      final path = Path()
        ..moveTo(pa.dx, pa.dy)
        ..quadraticBezierTo(ctrl.dx, ctrl.dy, pb.dx, pb.dy);
      canvas.drawPath(path, paint);

      if (showEdgeLabels) {
        final labelPos = _quadBezier(pa, ctrl, pb, 0.5);
        _drawEdgeLabel(canvas, labelPos, edge.relationType);
      }
    }
  }

  void _drawEdgeLabel(Canvas canvas, Offset pos, String label) {
    final tp = TextPainter(
      text: TextSpan(
        text: label,
        style: const TextStyle(
          fontSize: 9,
          color: Color(0xFF666666),
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    // Background rect
    final rect = Rect.fromCenter(
      center: pos,
      width: tp.width + 6,
      height: tp.height + 4,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(2)),
      Paint()..color = Colors.white.withValues(alpha: 0.85),
    );
    tp.paint(canvas, Offset(pos.dx - tp.width / 2, pos.dy - tp.height / 2));
  }

  void _drawNode(Canvas canvas, _GNode node) {
    final pos = positions[node.id];
    if (pos == null) return;

    final isSelected = node.id == selectedNodeId;
    final isHovered = node.id == hoveredNodeId;
    final color = nodeColorFn(node);
    // Dynamic radius: statement citations = full size, internal-only = 1/5 size
    final growthPerCite = (node.role == 'evidence' && node.citedInStatement == 0) ? 0.8 : 4.0;
    final double fullRadius = _nodeRadius + (node.statementCount * growthPerCite).clamp(0.0, 40.0);
    final double baseRadius;
    if (node.role == 'evidence' && node.uncitedRounds > 0) {
      baseRadius = (fullRadius - node.uncitedRounds * 4.0).clamp(4.0, fullRadius);
    } else {
      baseRadius = fullRadius;
    }
    final radius = isHovered ? baseRadius + 2 : baseRadius;

    // Fill
    canvas.drawCircle(
      pos,
      radius,
      Paint()..color = color,
    );

    // Border
    if (isSelected) {
      canvas.drawCircle(
        pos,
        radius + 2,
        Paint()
          ..color = Colors.orange
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3,
      );
    } else {
      canvas.drawCircle(
        pos,
        radius,
        Paint()
          ..color = const Color(0xFFE5E5E5)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1,
      );
    }

    // Label below node
    final label =
        node.name.length > 10 ? '${node.name.substring(0, 10)}...' : node.name;
    final tp = _labelCache.putIfAbsent(label, () => TextPainter(
        text: TextSpan(text: label, style: const TextStyle(fontSize: 10, color: Colors.black)),
        textDirection: TextDirection.ltr,
    )..layout());
    tp.paint(
      canvas,
      Offset(pos.dx - tp.width / 2, pos.dy + radius + 4),
    );

    // Team badge below label
    if (node.team.isNotEmpty) {
      final badge = node.team == 'team_a' ? '[A]' : node.team == 'team_b' ? '[B]' : '[J]';
      final badgeColor = node.team == 'team_a' ? const Color(0xFF4A90D9) : node.team == 'team_b' ? const Color(0xFFE74C3C) : Colors.amber;
      final badgeTp = _labelCache.putIfAbsent('badge_$badge', () => TextPainter(
          text: TextSpan(text: badge, style: TextStyle(fontSize: 8, color: badgeColor, fontWeight: FontWeight.bold)),
          textDirection: TextDirection.ltr,
      )..layout());
      badgeTp.paint(canvas, Offset(pos.dx - badgeTp.width / 2, pos.dy + radius + 16));
    }
  }

  // -- helpers (color resolved via injected functions) --

  String _pairKey(String a, String b) {
    return a.compareTo(b) < 0 ? '$a|$b' : '$b|$a';
  }

  double _curvatureForIndex(int idx, int count) {
    if (count <= 1) return 0;
    final half = (idx + 1) ~/ 2;
    final sign = idx.isOdd ? 1.0 : -1.0;
    return sign * half * 40.0;
  }

  Offset _quadBezier(Offset a, Offset ctrl, Offset b, double t) {
    final u = 1.0 - t;
    return a * (u * u) + ctrl * (2 * u * t) + b * (t * t);
  }

  @override
  bool shouldRepaint(covariant _GraphPainter oldDelegate) {
    if (!layoutDone) return true;
    return oldDelegate.selectedNodeId != selectedNodeId
        || oldDelegate.selectedEdgeId != selectedEdgeId
        || oldDelegate.hoveredNodeId != hoveredNodeId
        || oldDelegate.showEdgeLabels != showEdgeLabels;
  }
}

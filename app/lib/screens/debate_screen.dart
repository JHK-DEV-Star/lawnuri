import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:file_picker/file_picker.dart';
import 'package:url_launcher/url_launcher.dart';

import '../providers/debate_provider.dart';
import '../providers/settings_provider.dart';
import '../models/debate.dart';
import '../models/agent.dart';
import '../models/verdict.dart';
import '../api/debate_api.dart';
import '../api/rag_api.dart';
import '../api/report_api.dart';
import '../widgets/debate_graph_widget.dart';
import '../l10n/app_strings.dart';

/// Real-time debate monitoring screen with timeline, graph placeholder,
/// user intervention controls, and pause/stop actions.
/// MiroFish light theme variant.
class DebateScreen extends ConsumerStatefulWidget {
  final String debateId;
  const DebateScreen({super.key, required this.debateId});

  @override
  ConsumerState<DebateScreen> createState() => _DebateScreenState();
}

class _DebateScreenState extends ConsumerState<DebateScreen>
    with SingleTickerProviderStateMixin {
  final ScrollController _timelineScrollCtrl = ScrollController();
  final TextEditingController _interventionCtrl = TextEditingController();

  String _interventionTeam = 'team_a';
  int _previousLogLength = 0;

  // Pause overlay state.
  // _pauseModelOverride removed — model changes now happen in agent detail dialog.

  // Graph data
  List<Map<String, dynamic>> _graphNodes = [];
  List<Map<String, dynamic>> _graphEdges = [];
  List<Map<String, dynamic>> _interventions = [];
  int _graphTabIndex = 0;  // 0=graph, 1=interventions
  Timer? _graphRefreshTimer;

  // Report summary data (loaded when debate completes)
  Map<String, dynamic>? _reportSummary;

  // Auto-start step tracking
  /// Auto-start pipeline step: 0=idle, 1=analyzing, 2=generating agents, 3=starting debate, 4=running.
  int _autoStep = 0;
  String? _autoError;

  // Ctrl+F search
  bool _showSearch = false;
  String _searchQuery = '';
  int _currentMatchIndex = 0;
  final List<_SearchMatch> _searchMatches = [];
  final Map<int, GlobalKey> _logCardKeys = {};
  final Set<String> _openedToggles = {};  // toggles opened by search navigation
  final Map<int, bool> _discussionExpanded = {};  // manual toggle state per card
  final Map<int, bool> _qaExpanded = {};
  final Map<int, bool> _cardExpanded = {};  // cardIndex → expanded state
  final Map<int, ExpansionTileController> _cardControllers = {};
  GlobalKey? _currentMatchKey;  // key of the currently focused match
  final Map<String, int> _blockStartIndex = {};  // precomputed block_key → global match start index
  final GlobalKey _verdictSectionKey = GlobalKey();
  final TextEditingController _searchCtrl = TextEditingController();
  final FocusNode _searchFocusNode = FocusNode();

  // Pulse animation for running status.
  late final AnimationController _pulseController;
  late final Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    );
    _pulseAnimation = Tween<double>(begin: 0.5, end: 1.0).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    // Periodic graph refresh (every 5 seconds while running)
    _graphRefreshTimer = Timer.periodic(const Duration(seconds: 5), (_) {
      final status = ref.read(debateProvider).status;
      if (status == 'running' || status == 'extended') {
        _loadGraphData();
      }
    });

    // Load the debate, then auto-start if in 'created' state.
    Future.microtask(() async {
      await ref.read(debateProvider.notifier).loadDebate(widget.debateId);
      ref.read(settingsProvider.notifier).loadModels();
      _loadGraphData();
      // Auto-start flow if debate is freshly created
      final status = ref.read(debateProvider).status;
      if (status == 'created' || status == '') {
        await _autoStartFlow();
      }
    });
  }

  /// Automatically run analyze → generate agents → start debate.
  Future<void> _autoStartFlow() async {
    final notifier = ref.read(debateProvider.notifier);

    // Step 1: Analyze
    if (!mounted) return;
    setState(() { _autoStep = 1; _autoError = null; });
    await notifier.analyzeDebate();
    if (ref.read(debateProvider).error != null) {
      if (mounted) setState(() => _autoError = ref.read(debateProvider).error);
      return;
    }

    // Step 2: Generate agents
    if (!mounted) return;
    setState(() => _autoStep = 2);
    await notifier.generateAgents();
    if (ref.read(debateProvider).error != null) {
      if (mounted) setState(() => _autoError = ref.read(debateProvider).error);
      return;
    }

    // Reload agents into state
    await notifier.loadAgents();

    // Step 3: Start debate
    if (!mounted) return;
    setState(() => _autoStep = 3);
    await notifier.startDebate();
    if (ref.read(debateProvider).error != null) {
      if (mounted) setState(() => _autoError = ref.read(debateProvider).error);
      return;
    }

    if (mounted) setState(() => _autoStep = 4);
    _loadGraphData();
  }

  Future<void> _loadGraphData() async {
    try {
      final data = await DebateApi().getDebateGraph(widget.debateId);
      if (mounted) {
        final rawNodes = (data['nodes'] as List?) ?? [];
        final rawEdges = (data['edges'] as List?) ?? [];
        final rawInterventions = (data['interventions'] as List?) ?? [];
        setState(() {
          _graphNodes = rawNodes.map<Map<String, dynamic>>((n) =>
            Map<String, dynamic>.from(n as Map)
          ).toList();
          _graphEdges = rawEdges.map<Map<String, dynamic>>((e) =>
            Map<String, dynamic>.from(e as Map)
          ).toList();
          final serverInterventions = rawInterventions.map<Map<String, dynamic>>((i) =>
            Map<String, dynamic>.from(i as Map)
          ).toList();
          // Preserve local-only interventions not yet reflected in server
          final localOnly = _interventions.where((iv) =>
            iv['_local'] == true &&
            !serverInterventions.any((sv) =>
              sv['content'] == iv['content'] &&
              sv['round'] == iv['round'] &&
              sv['target_team'] == iv['target_team'])
          ).toList();
          _interventions = [...serverInterventions, ...localOnly];
        });
      }
    } catch (e) {
      debugPrint('[LawNuri] Graph load error: $e');
    }
  }

  @override
  void dispose() {
    _graphRefreshTimer?.cancel();
    // Auto-pause if debate is running when leaving the screen
    final status = ref.read(debateProvider).status;
    if (status == 'running' || status == 'extended') {
      ref.read(debateProvider.notifier).pauseDebate();
    }
    _pulseController.dispose();
    _timelineScrollCtrl.dispose();
    _interventionCtrl.dispose();
    _searchCtrl.dispose();
    _searchFocusNode.dispose();
    _cardControllers.clear();
    super.dispose();
  }

  /// Scroll the timeline to the bottom when new entries are appended.
  void _maybeAutoScroll(int currentLogLength) {
    if (currentLogLength > _previousLogLength) {
      _previousLogLength = currentLogLength;
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_timelineScrollCtrl.hasClients) {
          _timelineScrollCtrl.animateTo(
            _timelineScrollCtrl.position.maxScrollExtent,
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOut,
          );
        }
      });
    }
  }

  /// Send user intervention hint or evidence to the target team.
  Future<void> _sendIntervention() async {
    final text = _interventionCtrl.text.trim();
    if (text.isEmpty) return;

    await ref.read(debateProvider.notifier).interrupt(
          targetTeam: _interventionTeam,
          content: text,
          type: 'hint',
        );

    _interventionCtrl.clear();
    // Add to local interventions list immediately
    setState(() {
      _interventions.add({
        'type': 'hint',
        'content': text,
        'round': ref.read(debateProvider).currentRound,
        'target_team': _interventionTeam,
        '_local': true,
      });
    });
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(S.get('intervention_sent'))),
      );
    }
  }

  /// Upload an evidence file during the debate.
  Future<void> _uploadEvidence() async {
    final result = await FilePicker.platform.pickFiles();
    if (result == null || result.files.isEmpty) return;

    final file = result.files.first;
    if (file.path == null) return;

    try {
      final ragApi = RagApi();
      await ragApi.uploadDocument(
        widget.debateId,
        _interventionTeam,
        File(file.path!),
      );
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Evidence "${file.name}" uploaded.')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Upload failed: $e')),
        );
      }
    }
  }

  /// Pause the running debate.
  Future<void> _pause() async {
    await ref.read(debateProvider.notifier).pauseDebate();
  }

  /// Resume the paused debate, optionally applying model overrides.
  Future<void> _resume() async {
    await ref.read(debateProvider.notifier).resumeDebate();
  }

  /// Stop the debate permanently.
  Future<void> _stop() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(S.get('stop_debate')),
        content:
            const Text('Are you sure you want to stop this debate permanently?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: FilledButton.styleFrom(backgroundColor: Colors.red),
            child: const Text('Stop'),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      await ref.read(debateProvider.notifier).stopDebate();
    }
  }

  /// Show dialog to extend debate by N rounds.
  Future<void> _showExtendDialog() async {
    int additionalRounds = 5;
    final result = await showDialog<int>(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDialogState) => AlertDialog(
          title: const Text('라운드 추가'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('추가할 라운드 수를 선택하세요:'),
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  IconButton(
                    icon: const Icon(Icons.remove_circle_outline),
                    onPressed: additionalRounds > 1
                        ? () => setDialogState(() => additionalRounds--)
                        : null,
                  ),
                  Text('$additionalRounds', style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
                  IconButton(
                    icon: const Icon(Icons.add_circle_outline),
                    onPressed: additionalRounds < 20
                        ? () => setDialogState(() => additionalRounds++)
                        : null,
                  ),
                ],
              ),
            ],
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('취소')),
            FilledButton(
              onPressed: () => Navigator.pop(ctx, additionalRounds),
              child: const Text('추가 및 재개'),
            ),
          ],
        ),
      ),
    );

    if (result != null && result > 0) {
      final notifier = ref.read(debateProvider.notifier);
      await notifier.extendDebate(additionalRounds: result);
      await notifier.resumeDebate();
    }
  }

  @override
  Widget build(BuildContext context) {
    final debateState = ref.watch(debateProvider);
    final log = debateState.log;

    // Pre-compute search matches with position tracking
    _searchMatches.clear();
    if (_searchQuery.isNotEmpty) {
      final q = _searchQuery.toLowerCase();
      for (int i = 0; i < log.length; i++) {
        final entry = log[i];

        // Main text matches (speaker + statement)
        final mainText = '${entry.speaker} ${entry.statement}'.toLowerCase();
        int idx = 0;
        while ((idx = mainText.indexOf(q, idx)) != -1) {
          _searchMatches.add(_SearchMatch(cardIndex: i));
          idx += q.length;
        }

        // Evidence chips excluded from search matches — they use plain Text()
        // without _highlightText(), so counting them causes index misalignment
        // and scroll/highlight failures.

        // Internal discussion matches (shared text extraction)
        for (final d in entry.internalDiscussion) {
          final dText = _discussionSearchText(d).toLowerCase();
          idx = 0;
          while ((idx = dText.indexOf(q, idx)) != -1) {
            _searchMatches.add(_SearchMatch(
                cardIndex: i, isInToggle: true, toggleType: 'discussion'));
            idx += q.length;
          }
        }
      }

      // Verdict reasoning matches (continuous index after log cards)
      final verdicts = ref.read(debateProvider).verdicts;
      for (int vi = 0; vi < verdicts.length; vi++) {
        final vText = verdicts[vi].reasoning.toLowerCase();
        int vidx = 0;
        while ((vidx = vText.indexOf(q, vidx)) != -1) {
          _searchMatches.add(_SearchMatch(
            cardIndex: log.length + vi,  // continuous with log cards
            isInToggle: false,
            toggleType: 'verdict',
          ));
          vidx += q.length;
        }
      }
    }

    // (match offsets precomputed via _precomputeBlockOffsets)

    // Only run pulse animation when debate is running.
    final status = debateState.status;
    if (status == 'running' && !_pulseController.isAnimating) {
      _pulseController.repeat(reverse: true);
    } else if (status != 'running' && _pulseController.isAnimating) {
      _pulseController.stop();
      _pulseController.value = 0;
    }

    // Auto-scroll on new entries.
    _maybeAutoScroll(log.length);

    // Load graph once if empty (timer handles periodic refresh while running)
    if (_graphNodes.isEmpty && debateState.status != 'created') {
      Future.microtask(() => _loadGraphData());
    }

    // Verdict is shown in step 04 of the debate screen (no separate page needed).

    return Shortcuts(
      shortcuts: {
        LogicalKeySet(LogicalKeyboardKey.control, LogicalKeyboardKey.keyF):
            const _OpenSearchIntent(),
        LogicalKeySet(LogicalKeyboardKey.escape):
            const _CloseSearchIntent(),
      },
      child: Actions(
        actions: {
          _OpenSearchIntent: CallbackAction<_OpenSearchIntent>(
            onInvoke: (_) {
              setState(() {
                _showSearch = true;
              });
              Future.delayed(const Duration(milliseconds: 100), () {
                _searchFocusNode.requestFocus();
              });
              return null;
            },
          ),
          _CloseSearchIntent: CallbackAction<_CloseSearchIntent>(
            onInvoke: (_) {
              if (_showSearch) {
                setState(() {
                  _showSearch = false;
                  _searchQuery = '';
                  _searchCtrl.clear();
                });
              }
              return null;
            },
          ),
        },
        child: Focus(
          autofocus: true,
          child: Scaffold(
      backgroundColor: Colors.white,
      appBar: _buildAppBar(debateState),
      body: Stack(
        children: [
          Column(
            children: [
              // Search bar (Ctrl+F)
              if (_showSearch)
                _buildSearchBar(),
              // Main content: graph + timeline side by side.
              Expanded(
                child: Row(
                  children: [
                    // Left panel -- debate graph placeholder.
                    Expanded(
                      flex: 6,
                      child: _buildGraphPanel(),
                    ),
                    // Vertical divider: light border instead of VerticalDivider.
                    Container(
                      width: 1,
                      color: const Color(0xFFEAEAEA),
                    ),
                    // Right panel -- debate timeline.
                    Expanded(
                      flex: 4,
                      child: _buildTimelinePanel(log),
                    ),
                  ],
                ),
              ),
              Container(
                height: 1,
                color: const Color(0xFFEAEAEA),
              ),
              // Bottom panel -- user intervention.
              _buildInterventionPanel(debateState),
            ],
          ),

          // Paused overlay.
          // Paused overlay removed — pause/resume is now a simple toggle button.
        ],
      ),
    ),  // Focus
    ),  // Actions
    ),  // Shortcuts
    );
  }

  // ---------------------------------------------------------------------------
  // Search bar (Ctrl+F)
  // ---------------------------------------------------------------------------

  Widget _buildSearchBar() {
    final totalMatches = _searchMatches.length;
    final displayIndex = totalMatches > 0 ? _currentMatchIndex + 1 : 0;

    return Container(
      height: 44,
      padding: const EdgeInsets.symmetric(horizontal: 12),
      decoration: const BoxDecoration(
        color: Color(0xFFF5F5F5),
        border: Border(bottom: BorderSide(color: Color(0xFFE5E5E5))),
      ),
      child: Row(
        children: [
          const Icon(Icons.search, size: 18, color: Color(0xFF666666)),
          const SizedBox(width: 8),
          Expanded(
            child: TextField(
              controller: _searchCtrl,
              focusNode: _searchFocusNode,
              style: const TextStyle(fontSize: 13),
              decoration: InputDecoration(
                hintText: S.get('search_debate'),
                border: InputBorder.none,
                enabledBorder: InputBorder.none,
                focusedBorder: InputBorder.none,
                isDense: true,
                contentPadding: const EdgeInsets.symmetric(vertical: 8),
                fillColor: Colors.transparent,
              ),
              onChanged: (v) => setState(() {
                _searchQuery = v;
                _currentMatchIndex = 0;
              }),
              onSubmitted: (_) => _goToNextMatch(),
            ),
          ),
          const SizedBox(width: 8),
          if (_searchQuery.isNotEmpty)
            Text(
              '$displayIndex/$totalMatches',
              style: const TextStyle(fontSize: 12, color: Color(0xFF999999)),
            ),
          const SizedBox(width: 4),
          // Previous match
          IconButton(
            icon: const Icon(Icons.keyboard_arrow_up, size: 20),
            onPressed: totalMatches > 0 ? _goToPrevMatch : null,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
            tooltip: S.get('previous_match'),
          ),
          // Next match
          IconButton(
            icon: const Icon(Icons.keyboard_arrow_down, size: 20),
            onPressed: totalMatches > 0 ? _goToNextMatch : null,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
            tooltip: S.get('next_match'),
          ),
          const SizedBox(width: 4),
          // Close
          IconButton(
            icon: const Icon(Icons.close, size: 16),
            onPressed: () => setState(() {
              _showSearch = false;
              _searchQuery = '';
              _searchCtrl.clear();
              _searchMatches.clear();
              _openedToggles.clear();
              _discussionExpanded.clear();
              _qaExpanded.clear();
              _currentMatchIndex = 0;
              _currentMatchKey = null;
            }),
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
          ),
        ],
      ),
    );
  }

  void _goToNextMatch() {
    if (_searchMatches.isEmpty) return;
    setState(() {
      _currentMatchIndex = (_currentMatchIndex + 1) % _searchMatches.length;
      _openToggleForCurrentMatch();
    });
    _scrollToCurrentMatch();
  }

  void _goToPrevMatch() {
    if (_searchMatches.isEmpty) return;
    setState(() {
      _currentMatchIndex = (_currentMatchIndex - 1 + _searchMatches.length) % _searchMatches.length;
      _openToggleForCurrentMatch();
    });
    _scrollToCurrentMatch();
  }

  void _openToggleForCurrentMatch() {
    if (_currentMatchIndex >= _searchMatches.length) return;
    final match = _searchMatches[_currentMatchIndex];

    // Always expand the card itself (use controller for programmatic expansion)
    _cardExpanded[match.cardIndex] = true;
    final ctrl = _cardControllers[match.cardIndex];
    if (ctrl != null) {
      try { ctrl.expand(); } catch (_) {}
    }

    // Expand inner toggle if match is inside one
    if (match.isInToggle) {
      if (match.toggleType == 'discussion') {
        _discussionExpanded[match.cardIndex] = true;
      } else if (match.toggleType == 'qa_answer') {
        _qaExpanded[match.cardIndex] = true;
      }
    }
  }

  void _scrollToCurrentMatch() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_currentMatchKey?.currentContext != null) {
        Scrollable.ensureVisible(
          _currentMatchKey!.currentContext!,
          alignment: 0.5,
          duration: const Duration(milliseconds: 300),
        );
      }
    });
  }

  // ---------------------------------------------------------------------------
  // App bar
  // ---------------------------------------------------------------------------

  PreferredSizeWidget _buildAppBar(DebateState debateState) {
    return AppBar(
      leading: IconButton(
        icon: const Icon(Icons.home_outlined),
        tooltip: 'Home',
        onPressed: () {
          ref.read(debateProvider.notifier).reset();
          context.go('/');
        },
      ),
      backgroundColor: Colors.white,
      foregroundColor: Colors.black87,
      elevation: 0,
      bottom: PreferredSize(
        preferredSize: const Size.fromHeight(1),
        child: Container(
          height: 1,
          color: const Color(0xFFEAEAEA),
        ),
      ),
      title: Text(
        '${S.get('debate_title')} - ${S.get('round_label')} ${debateState.currentRound}/${debateState.maxRounds}',
        style: const TextStyle(color: Colors.black87),
      ),
      actions: [
        // Status badge.
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8),
          child: _buildStatusChip(debateState.status),
        ),
        // Pause/Resume toggle button.
        if (debateState.status == 'running')
          IconButton(
            icon: const Icon(Icons.pause_circle_outline),
            tooltip: 'Pause',
            color: Colors.black87,
            onPressed: _pause,
          )
        else if (debateState.status == 'paused' || debateState.status == 'stopped')
          IconButton(
            icon: const Icon(Icons.play_circle_outline),
            tooltip: 'Resume',
            color: Colors.green,
            onPressed: _resume,
          ),
        // Stop button.
        if (debateState.status == 'running' ||
            debateState.status == 'paused')
          IconButton(
            icon: const Icon(Icons.stop_circle_outlined),
            tooltip: 'Stop',
            color: Colors.black87,
            onPressed: _stop,
          ),
        // Extend rounds button (for stopped/completed debates).
        if (debateState.status == 'stopped' ||
            debateState.status == 'completed' ||
            (debateState.status == 'running' && debateState.currentRound >= debateState.maxRounds && debateState.maxRounds > 0))
          TextButton.icon(
            onPressed: _showExtendDialog,
            icon: const Icon(Icons.add_circle_outline, size: 18),
            label: Text(S.get('add_rounds')),
            style: TextButton.styleFrom(foregroundColor: Colors.black87),
          ),
      ],
    );
  }

  /// Build a status chip with the appropriate color and pulse animation for running.
  Widget _buildStatusChip(String status) {
    final color = _statusColor(status);
    final chip = Chip(
      label: Text(
        status.toUpperCase(),
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.bold,
          color: color,
        ),
      ),
      backgroundColor: color.withOpacity(0.12),
      side: BorderSide(color: color.withOpacity(0.3)),
    );

    if (status == 'running') {
      return AnimatedBuilder(
        animation: _pulseAnimation,
        builder: (context, child) {
          return Opacity(
            opacity: _pulseAnimation.value,
            child: child,
          );
        },
        child: chip,
      );
    }

    return chip;
  }

  /// Return a color for the status badge.
  Color _statusColor(String status) {
    switch (status) {
      case 'running':
        return const Color(0xFFFF5722); // orange
      case 'paused':
        return Colors.orange;
      case 'finished':
      case 'completed':
        return const Color(0xFF4CAF50); // green
      case 'stopped':
      case 'error':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  // ---------------------------------------------------------------------------
  // Graph panel (placeholder)
  // ---------------------------------------------------------------------------

  Widget _buildGraphPanel() {
    final ds = ref.read(settingsProvider).debateSettings;
    final tA = ds['team_a_name'] as String? ?? 'Team A';
    final tB = ds['team_b_name'] as String? ?? 'Team B';
    if (_graphNodes.isEmpty) {
      return Container(
        decoration: BoxDecoration(
          color: Colors.white,
          border: Border.all(color: const Color(0xFFE5E5E5)),
        ),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.hub, size: 48, color: Color(0xFFCCCCCC)),
              const SizedBox(height: 12),
              Text(S.get('graph_placeholder'),
                  style: TextStyle(color: Color(0xFF999999), fontSize: 13)),
              const SizedBox(height: 8),
              TextButton.icon(
                onPressed: _loadGraphData,
                icon: const Icon(Icons.refresh, size: 16),
                label: Text(S.get('refresh')),
              ),
            ],
          ),
        ),
      );
    }
    final debateState = ref.read(debateProvider);
    final analysis = debateState.analysis;
    final opinionA = analysis?.opinionA ?? '';
    final opinionB = analysis?.opinionB ?? '';

    return Column(
      children: [
        // Team legend
        if (opinionA.isNotEmpty || opinionB.isNotEmpty)
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: const BoxDecoration(
              border: Border(bottom: BorderSide(color: Color(0xFFE5E5E5))),
            ),
            child: Wrap(
              spacing: 16,
              runSpacing: 4,
              children: [
                _teamLegendItem(Colors.blue, S.teamName('team_a', teamAName: tA, teamBName: tB), opinionA),
                _teamLegendItem(Colors.red, S.teamName('team_b', teamAName: tA, teamBName: tB), opinionB),
                _teamLegendItem(Colors.orange, S.get('judges'), S.get('neutral_eval')),
              ],
            ),
          ),
        // Tab bar
        Container(
          decoration: const BoxDecoration(
            border: Border(bottom: BorderSide(color: Color(0xFFE5E5E5))),
          ),
          child: Row(children: [
            _graphTabButton(0, Icons.hub, S.get('graph_tab')),
            _graphTabButton(1, Icons.person_pin, S.get('interventions_tab')),
          ]),
        ),
        // Tab content
        Expanded(
          child: _graphTabIndex == 0
              ? DebateGraphWidget(
                  nodes: _graphNodes,
                  edges: _graphEdges,
                  teamAName: tA,
                  teamBName: tB,
                )
              : _buildInterventionsTab(),
        ),
      ],
    );
  }

  Widget _graphTabButton(int index, IconData icon, String label) {
    final selected = _graphTabIndex == index;
    return InkWell(
      onTap: () => setState(() => _graphTabIndex = index),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          border: Border(
            bottom: BorderSide(
              color: selected ? Colors.blue : Colors.transparent,
              width: 2,
            ),
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: selected ? Colors.blue : Colors.grey),
            const SizedBox(width: 4),
            Text(label, style: TextStyle(
              fontSize: 12,
              color: selected ? Colors.blue : Colors.grey,
              fontWeight: selected ? FontWeight.bold : FontWeight.normal,
            )),
          ],
        ),
      ),
    );
  }

  Widget _buildInterventionsTab() {
    if (_interventions.isEmpty) {
      return Center(
        child: Text(S.get('no_interventions'),
          style: const TextStyle(color: Color(0xFF999999), fontSize: 13)),
      );
    }
    return ListView.separated(
      padding: const EdgeInsets.all(12),
      itemCount: _interventions.length,
      separatorBuilder: (_, __) => const Divider(height: 1),
      itemBuilder: (ctx, i) {
        final iv = _interventions[i];
        final content = iv['content'] as String? ?? '';
        final round = iv['round'] as int? ?? 0;
        final team = iv['target_team'] as String? ?? '';
        return ListTile(
          dense: true,
          leading: const Icon(
            Icons.lightbulb_outline,
            size: 20,
            color: Colors.orange,
          ),
          title: Text(content,
            style: const TextStyle(fontSize: 12, color: Colors.black87),
            maxLines: 3, overflow: TextOverflow.ellipsis),
          subtitle: Text(
            'Round $round · $team',
            style: const TextStyle(fontSize: 10, color: Colors.grey)),
        );
      },
    );
  }

  Widget _teamLegendItem(Color color, String team, String opinion) {
    final shortOpinion = opinion.length > 30
        ? '${opinion.substring(0, 30)}...'
        : opinion;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 8, height: 8,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 4),
        Text('$team: ', style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold)),
        Text(shortOpinion, style: const TextStyle(fontSize: 11, color: Color(0xFF666666))),
      ],
    );
  }

  // ---------------------------------------------------------------------------
  // Timeline panel
  // ---------------------------------------------------------------------------

  Widget _buildAutoStartProgress() {
    final steps = [
      '상황 분석',
      '에이전트 생성',
      '토론 시작',
    ];
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            for (int i = 0; i < steps.length; i++)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8),
                child: Row(
                  children: [
                    if (i + 1 < _autoStep)
                      const Icon(Icons.check_circle, color: Color(0xFF4CAF50), size: 20)
                    else if (i + 1 == _autoStep && _autoError == null)
                      const SizedBox(
                        width: 20, height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFFFF4500)),
                      )
                    else if (i + 1 == _autoStep && _autoError != null)
                      const Icon(Icons.error, color: Colors.red, size: 20)
                    else
                      const Icon(Icons.circle_outlined, color: Color(0xFFCCCCCC), size: 20),
                    const SizedBox(width: 12),
                    Text(
                      steps[i],
                      style: TextStyle(
                        fontSize: 14,
                        color: i + 1 <= _autoStep ? Colors.black87 : const Color(0xFF999999),
                        fontWeight: i + 1 == _autoStep ? FontWeight.w600 : FontWeight.normal,
                      ),
                    ),
                    if (i + 1 == _autoStep && _autoError == null)
                      const Padding(
                        padding: EdgeInsets.only(left: 8),
                        child: Text('진행 중...', style: TextStyle(color: Color(0xFF999999), fontSize: 12)),
                      ),
                  ],
                ),
              ),
            if (_autoError != null) ...[
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red.withValues(alpha: 0.05),
                  border: Border.all(color: Colors.red.withValues(alpha: 0.3)),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(_autoError!, style: const TextStyle(color: Colors.red, fontSize: 12)),
                    const SizedBox(height: 8),
                    TextButton(
                      onPressed: () {
                        setState(() => _autoError = null);
                        _autoStartFlow();
                      },
                      child: Text(S.get('retry')),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildTimelinePanel(List<DebateLogEntry> log) {
    final ds = ref.read(settingsProvider).debateSettings;
    final tA = ds['team_a_name'] as String? ?? 'Team A';
    final tB = ds['team_b_name'] as String? ?? 'Team B';
    _currentMatchKey = null;

    // Show auto-start steps if in progress
    if (_autoStep > 0 && _autoStep < 4) {
      return _buildAutoStartProgress();
    }

    final debateState = ref.watch(debateProvider);
    final analysis = debateState.analysis;
    final agents = debateState.agents;
    final verdicts = debateState.verdicts;
    final status = debateState.status;

    // Precompute block offsets for search (render-independent)
    _precomputeBlockOffsets(log, verdicts);

    // Determine step statuses
    final step1Status = analysis != null ? 'completed' : 'pending';
    final step2Status = agents.isNotEmpty ? 'completed' : 'pending';
    final step3Status = ['completed', 'finished', 'stopped'].contains(status)
        ? 'completed'
        : (status == 'running' ? 'processing' : (log.isNotEmpty ? 'completed' : 'pending'));
    final step4Status = verdicts.isNotEmpty ? 'completed' : 'pending';

    return Container(
      color: Colors.white,
      child: SingleChildScrollView(
        controller: _timelineScrollCtrl,
        padding: const EdgeInsets.symmetric(horizontal: 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
              const SizedBox(height: 8),

              // Step 01 - Topic Analysis
              _buildStepCard(
                number: '01',
                title: S.get('topic_analysis'),
                status: step1Status,
                content: analysis != null
                    ? Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          SelectableText(analysis.topic,
                              style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
                          const SizedBox(height: 12),
                          // Team A opinion
                          Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Container(
                                width: 8, height: 8,
                                margin: const EdgeInsets.only(top: 5),
                                decoration: const BoxDecoration(
                                    color: Colors.blue, shape: BoxShape.circle),
                              ),
                              const SizedBox(width: 8),
                              Expanded(
                                child: SelectableText(analysis.opinionA,
                                    style: const TextStyle(fontSize: 13, height: 1.5)),
                              ),
                            ],
                          ),
                          const SizedBox(height: 8),
                          // Team B opinion
                          Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Container(
                                width: 8, height: 8,
                                margin: const EdgeInsets.only(top: 5),
                                decoration: const BoxDecoration(
                                    color: Colors.red, shape: BoxShape.circle),
                              ),
                              const SizedBox(width: 8),
                              Expanded(
                                child: SelectableText(analysis.opinionB,
                                    style: const TextStyle(fontSize: 13, height: 1.5)),
                              ),
                            ],
                          ),
                          const SizedBox(height: 12),
                          // Key issues
                          ...analysis.keyIssues.map((issue) => Padding(
                                padding: const EdgeInsets.only(bottom: 4),
                                child: Row(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    const Text('  \u2022 ',
                                        style: TextStyle(fontSize: 13, color: Color(0xFF666666))),
                                    Expanded(
                                      child: SelectableText(issue,
                                          style: const TextStyle(
                                              fontSize: 12, color: Color(0xFF666666), height: 1.4)),
                                    ),
                                  ],
                                ),
                              )),
                          // Team A cautions
                          if (analysis.teamACautions.isNotEmpty) ...[
                            const SizedBox(height: 12),
                            Row(children: [
                              Container(width: 8, height: 8,
                                  decoration: const BoxDecoration(color: Colors.blue, shape: BoxShape.circle)),
                              const SizedBox(width: 6),
                              Text('${S.teamName('team_a', teamAName: tA, teamBName: tB)} ${S.get("cautions")}',
                                  style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
                            ]),
                            const SizedBox(height: 4),
                            ...analysis.teamACautions.map((c) => Padding(
                                  padding: const EdgeInsets.only(bottom: 2, left: 14),
                                  child: SelectableText('\u26A0 $c',
                                      style: const TextStyle(fontSize: 11, color: Color(0xFFE65100), height: 1.4)),
                                )),
                          ],
                          // Team B cautions
                          if (analysis.teamBCautions.isNotEmpty) ...[
                            const SizedBox(height: 8),
                            Row(children: [
                              Container(width: 8, height: 8,
                                  decoration: const BoxDecoration(color: Colors.red, shape: BoxShape.circle)),
                              const SizedBox(width: 6),
                              Text('${S.teamName('team_b', teamAName: tA, teamBName: tB)} ${S.get("cautions")}',
                                  style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
                            ]),
                            const SizedBox(height: 4),
                            ...analysis.teamBCautions.map((c) => Padding(
                                  padding: const EdgeInsets.only(bottom: 2, left: 14),
                                  child: SelectableText('\u26A0 $c',
                                      style: const TextStyle(fontSize: 11, color: Color(0xFFE65100), height: 1.4)),
                                )),
                          ],
                          // Parties
                          if (analysis.parties.isNotEmpty) ...[
                            const SizedBox(height: 16),
                            const Divider(height: 1, color: Color(0xFFEAEAEA)),
                            const SizedBox(height: 12),
                            Text(S.get('parties'),
                                style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: Color(0xFF333333))),
                            const SizedBox(height: 6),
                            ...analysis.parties.map((p) => Padding(
                              padding: const EdgeInsets.only(bottom: 4),
                              child: Row(children: [
                                Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                  decoration: BoxDecoration(
                                    color: const Color(0xFFF5F5F5),
                                    borderRadius: BorderRadius.circular(3),
                                  ),
                                  child: SelectableText(p['role']?.toString() ?? '',
                                      style: const TextStyle(fontSize: 10, color: Color(0xFF666666))),
                                ),
                                const SizedBox(width: 8),
                                Expanded(child: SelectableText(
                                  '${p['name'] ?? ''} — ${p['description'] ?? ''}',
                                  style: const TextStyle(fontSize: 12, height: 1.4),
                                )),
                              ]),
                            )),
                          ],
                          // Timeline (collapsed)
                          if (analysis.timeline.isNotEmpty) ...[
                            const SizedBox(height: 12),
                            Theme(
                              data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
                              child: ExpansionTile(
                                tilePadding: EdgeInsets.zero,
                                title: Text('Timeline (${analysis.timeline.length})',
                                    style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700)),
                                initiallyExpanded: _searchQuery.isNotEmpty &&
                                    analysis.timeline.any((t) =>
                                        t.values.any((v) => v.toString().toLowerCase().contains(_searchQuery.toLowerCase()))),
                                children: analysis.timeline.map((t) => Padding(
                                  padding: const EdgeInsets.only(bottom: 6, left: 8),
                                  child: Row(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      SizedBox(
                                        width: 70,
                                        child: SelectableText(t['date']?.toString() ?? '?',
                                            style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: Color(0xFF666666))),
                                      ),
                                      Expanded(child: Column(
                                        crossAxisAlignment: CrossAxisAlignment.start,
                                        children: [
                                          SelectableText(t['action']?.toString() ?? '',
                                              style: const TextStyle(fontSize: 11, height: 1.3)),
                                          if (t['significance'] != null)
                                            SelectableText(t['significance'].toString(),
                                                style: const TextStyle(fontSize: 10, color: Color(0xFF999999), height: 1.3)),
                                        ],
                                      )),
                                    ],
                                  ),
                                )).toList(),
                              ),
                            ),
                          ],
                          // Key Facts (collapsed)
                          if (analysis.keyFacts.isNotEmpty) ...[
                            Theme(
                              data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
                              child: ExpansionTile(
                                tilePadding: EdgeInsets.zero,
                                title: Text('Key Facts (${analysis.keyFacts.length})',
                                    style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700)),
                                initiallyExpanded: _searchQuery.isNotEmpty &&
                                    analysis.keyFacts.any((f) =>
                                        f.values.any((v) => v.toString().toLowerCase().contains(_searchQuery.toLowerCase()))),
                                children: analysis.keyFacts.map((f) {
                                  final disputed = f['disputed'] == true;
                                  final importance = f['importance']?.toString() ?? '';
                                  return Padding(
                                    padding: const EdgeInsets.only(bottom: 4, left: 8),
                                    child: Row(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Container(
                                          margin: const EdgeInsets.only(top: 3, right: 6),
                                          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                                          decoration: BoxDecoration(
                                            color: importance == 'critical' ? const Color(0xFFFFEBEE)
                                                : importance == 'high' ? const Color(0xFFFFF3E0)
                                                : const Color(0xFFF5F5F5),
                                            borderRadius: BorderRadius.circular(2),
                                          ),
                                          child: Text(importance.toUpperCase(),
                                              style: TextStyle(fontSize: 9, fontWeight: FontWeight.w700,
                                                  color: importance == 'critical' ? Colors.red : const Color(0xFF666666))),
                                        ),
                                        if (disputed)
                                          Container(
                                            margin: const EdgeInsets.only(top: 3, right: 6),
                                            padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                                            decoration: BoxDecoration(
                                              color: const Color(0xFFFCE4EC),
                                              borderRadius: BorderRadius.circular(2),
                                            ),
                                            child: Text(S.get('disputed').toUpperCase(),
                                                style: const TextStyle(fontSize: 9, fontWeight: FontWeight.w700, color: Colors.red)),
                                          ),
                                        Expanded(child: SelectableText(f['fact']?.toString() ?? '',
                                            style: const TextStyle(fontSize: 11, height: 1.3))),
                                      ],
                                    ),
                                  );
                                }).toList(),
                              ),
                            ),
                          ],
                          // Causal Chain
                          if (analysis.causalChain.isNotEmpty) ...[
                            const SizedBox(height: 8),
                            Text(S.get('causal_chain'),
                                style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: Color(0xFF333333))),
                            const SizedBox(height: 4),
                            ...analysis.causalChain.map((c) => Padding(
                              padding: const EdgeInsets.only(bottom: 2),
                              child: SelectableText(c, style: const TextStyle(fontSize: 11, color: Color(0xFF666666), height: 1.4)),
                            )),
                          ],
                          // Missing Information
                          if (analysis.missingInformation.isNotEmpty) ...[
                            const SizedBox(height: 12),
                            Container(
                              padding: const EdgeInsets.all(8),
                              decoration: BoxDecoration(
                                color: const Color(0xFFFFF8E1),
                                borderRadius: BorderRadius.circular(4),
                                border: Border.all(color: const Color(0xFFFFE082)),
                              ),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Row(children: [
                                    const Icon(Icons.info_outline, size: 14, color: Color(0xFFF57C00)),
                                    const SizedBox(width: 4),
                                    Text(S.get('missing_info'),
                                        style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: Color(0xFFF57C00))),
                                  ]),
                                  const SizedBox(height: 4),
                                  ...analysis.missingInformation.map((m) => Padding(
                                    padding: const EdgeInsets.only(bottom: 2),
                                    child: SelectableText('\u2022 $m',
                                        style: const TextStyle(fontSize: 11, color: Color(0xFF666666))),
                                  )),
                                ],
                              ),
                            ),
                          ],
                        ],
                      )
                    : Text(S.get('analysis_pending'),
                        style: TextStyle(fontSize: 13, color: Color(0xFF999999))),
              ),

              // Step 02 - Agents
              _buildStepCard(
                number: '02',
                title: S.get('agents'),
                status: step2Status,
                content: agents.isNotEmpty
                    ? Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // Team A agents
                          ..._buildAgentGroup(S.teamName('team_a', teamAName: tA, teamBName: tB), agents.where((a) => a.team == 'team_a').toList()),
                          const SizedBox(height: 8),
                          // Team B agents
                          ..._buildAgentGroup(S.teamName('team_b', teamAName: tA, teamBName: tB), agents.where((a) => a.team == 'team_b').toList()),
                          const SizedBox(height: 8),
                          // Judge agents
                          ..._buildAgentGroup(S.get('judges_label'), agents.where((a) => a.team != 'team_a' && a.team != 'team_b').toList()),
                        ],
                      )
                    : Text(S.get('agents_pending'),
                        style: TextStyle(fontSize: 13, color: Color(0xFF999999))),
              ),

              // Step 03 - Debate Progress
              _buildStepCard(
                number: '03',
                title: S.get('debate_progress'),
                status: step3Status,
                content: _buildDebateProgressContent(log, debateState),
              ),

              // Step 04 - Verdict
              Container(
                key: _verdictSectionKey,
                child: _buildStepCard(
                number: '04',
                title: S.get('verdict'),
                status: step4Status,
                content: verdicts.isNotEmpty
                    ? _buildVerdictContent(verdicts, logLength: log.length)
                    : Text(S.get('after_debate_verdict'),
                        style: const TextStyle(fontSize: 13, color: Color(0xFF999999))),
              )),

              // Step 05 - Report
              _buildStepCard(
                number: '05',
                title: S.get('report'),
                status: ['completed', 'stopped', 'finished'].contains(debateState.status) && verdicts.isNotEmpty
                    ? 'completed' : 'pending',
                content: ['completed', 'stopped', 'finished'].contains(debateState.status) && verdicts.isNotEmpty
                    ? _buildReportSummary(debateState.debateId ?? widget.debateId)
                    : Text(S.get('after_debate_report'),
                        style: const TextStyle(fontSize: 13, color: Color(0xFF999999))),
              ),

              const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Step card & helper widgets
  // ---------------------------------------------------------------------------

  Widget _buildVerdictContent(List<Verdict> verdicts, {int logLength = 0}) {
    final ds = ref.read(settingsProvider).debateSettings;
    final tA = ds['team_a_name'] as String? ?? 'Team A';
    final tB = ds['team_b_name'] as String? ?? 'Team B';
    // Determine majority verdict
    final verdictCounts = <String, int>{};
    for (final v in verdicts) {
      verdictCounts[v.verdict] = (verdictCounts[v.verdict] ?? 0) + 1;
    }
    final majorityVerdict = verdictCounts.entries
        .reduce((a, b) => a.value >= b.value ? a : b)
        .key;
    final majorityLabel = majorityVerdict == 'team_a'
        ? '${S.teamName('team_a', teamAName: tA, teamBName: tB)} 승리'
        : majorityVerdict == 'team_b'
            ? '${S.teamName('team_b', teamAName: tA, teamBName: tB)} 승리'
            : '무승부';
    final majorityColor = majorityVerdict == 'team_a'
        ? Colors.blue
        : majorityVerdict == 'team_b'
            ? Colors.red
            : Colors.grey;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Overall result banner
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: majorityColor.withAlpha(25),
            border: Border.all(color: majorityColor.withAlpha(77)),
            borderRadius: BorderRadius.circular(4),
          ),
          child: Row(
            children: [
              Icon(Icons.gavel, size: 20, color: majorityColor),
              const SizedBox(width: 8),
              Text(
                '최종 판결: $majorityLabel',
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w700,
                  color: majorityColor,
                ),
              ),
              const Spacer(),
              Text(
                '${verdictCounts[majorityVerdict]}/${verdicts.length} 심판',
                style: TextStyle(fontSize: 12, color: majorityColor),
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        // Individual judge verdicts
        ...verdicts.asMap().entries.map((ve) {
          final vi = ve.key;
          final v = ve.value;
          final vColor = v.verdict == 'team_a'
              ? Colors.blue
              : v.verdict == 'team_b'
                  ? Colors.red
                  : Colors.grey;
          return Container(
            margin: const EdgeInsets.only(bottom: 8),
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              border: Border.all(color: const Color(0xFFE5E5E5)),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Judge name + verdict badge + confidence
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: const Color(0xFFF5A623).withAlpha(30),
                        borderRadius: BorderRadius.circular(2),
                      ),
                      child: Text(v.judgeName.isNotEmpty ? v.judgeName : v.judgeId,
                          style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
                    ),
                    const SizedBox(width: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: vColor.withAlpha(30),
                        border: Border.all(color: vColor.withAlpha(77)),
                        borderRadius: BorderRadius.circular(2),
                      ),
                      child: Text(v.verdictLabel,
                          style: TextStyle(fontSize: 11, color: vColor, fontWeight: FontWeight.w600)),
                    ),
                    const SizedBox(width: 8),
                    Text('신뢰도: ${(v.confidence * 100).toInt()}%',
                        style: const TextStyle(fontSize: 11, color: Color(0xFF999999))),
                  ],
                ),
                if (v.reasoning.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  _buildRichStatement(v.reasoning, const [],
                      startIndex: _blockStartIndex['verdict_$vi'] ?? 0),
                ],
                if (v.teamAScore.isNotEmpty || v.teamBScore.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  _buildScoreBars(v.teamAScore, v.teamBScore),
                ],
              ],
            ),
          );
        }),
      ],
    );
  }

  Widget _buildScoreBars(Map<String, dynamic> teamA, Map<String, dynamic> teamB) {
    final categories = <String>{...teamA.keys, ...teamB.keys};
    if (categories.isEmpty) return const SizedBox.shrink();
    return Column(
      children: categories.map((cat) {
        final aScore = (teamA[cat] as num?)?.toDouble() ?? 50;
        final bScore = (teamB[cat] as num?)?.toDouble() ?? 50;
        final total = aScore + bScore;
        final aRatio = total > 0 ? aScore / total : 0.5;
        return Padding(
          padding: const EdgeInsets.only(bottom: 4),
          child: Row(
            children: [
              SizedBox(
                width: 130,
                child: Text(
                  cat.replaceAll('_', ' ').replaceAllMapped(
                    RegExp(r'(^|\s)\w'), (m) => m.group(0)!.toUpperCase()),
                  style: const TextStyle(fontSize: 10, color: Color(0xFF999999)),
                ),
              ),
              const SizedBox(width: 4),
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(2),
                  child: SizedBox(
                    height: 6,
                    child: Row(
                      children: [
                        Flexible(
                          flex: (aRatio * 100).round(),
                          child: Container(color: Colors.blue.withAlpha(180)),
                        ),
                        Flexible(
                          flex: ((1 - aRatio) * 100).round(),
                          child: Container(color: Colors.red.withAlpha(180)),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }

  Future<void> _regenerateReport(String debateId) async {
    try {
      await ReportApi().regenerateReport(debateId);
      if (mounted) context.go('/report/$debateId');
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Report regeneration failed: $e')),
        );
      }
    }
  }

  Widget _buildReportSummary(String debateId) {
    final ds = ref.read(settingsProvider).debateSettings;
    final tA = ds['team_a_name'] as String? ?? 'Team A';
    final tB = ds['team_b_name'] as String? ?? 'Team B';
    // Trigger load if not yet loaded
    if (_reportSummary == null) {
      Future.microtask(() async {
        try {
          final data = await ReportApi().getReport(debateId);
          if (mounted) setState(() => _reportSummary = data);
        } catch (_) {}
      });
      return const Center(child: Padding(
        padding: EdgeInsets.all(16),
        child: CircularProgressIndicator(strokeWidth: 2),
      ));
    }

    final report = _reportSummary!;
    final summary = report['executive_summary'];
    final summaryText = summary is Map
        ? (summary['summary'] ?? summary.toString())
        : summary?.toString() ?? '';
    final result = summary is Map ? (summary['result'] ?? '') : '';
    final analysis = report['argument_analysis'] as Map?;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Result badge
        if (result.toString().isNotEmpty)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            margin: const EdgeInsets.only(bottom: 8),
            decoration: BoxDecoration(
              color: result.toString().contains('team_a') ? Colors.blue.withAlpha(25)
                  : result.toString().contains('team_b') ? Colors.red.withAlpha(25)
                  : Colors.orange.withAlpha(25),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(result.toString(),
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 13,
                color: result.toString().contains('team_a') ? Colors.blue
                    : result.toString().contains('team_b') ? Colors.red
                    : Colors.orange)),
          ),
        // Summary text (truncated)
        if (summaryText.toString().isNotEmpty)
          Text(
            summaryText.toString().length > 300
                ? '${summaryText.toString().substring(0, 300)}...'
                : summaryText.toString(),
            style: const TextStyle(fontSize: 12, height: 1.5, color: Color(0xFF666666)),
          ),
        const SizedBox(height: 8),
        // Team evaluations
        if (analysis != null) ...[
          for (final team in ['team_a', 'team_b'])
            if (analysis[team] is Map) ...[
              Text('▪ ${S.teamName(team, teamAName: tA, teamBName: tB)}',
                style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold,
                  color: team == 'team_a' ? Colors.blue : Colors.red)),
              if ((analysis[team] as Map)['strongest'] != null)
                Text('  강점: ${((analysis[team] as Map)['strongest'] as List?)?.firstOrNull ?? ''}',
                  style: const TextStyle(fontSize: 11, color: Color(0xFF666666)),
                  maxLines: 1, overflow: TextOverflow.ellipsis),
              if ((analysis[team] as Map)['weakest'] != null)
                Text('  약점: ${((analysis[team] as Map)['weakest'] as List?)?.firstOrNull ?? ''}',
                  style: const TextStyle(fontSize: 11, color: Color(0xFF999999)),
                  maxLines: 1, overflow: TextOverflow.ellipsis),
              const SizedBox(height: 4),
            ],
        ],
        const SizedBox(height: 8),
        // Report buttons
        Row(children: [
          ElevatedButton.icon(
            onPressed: () => context.go('/report/$debateId'),
            icon: const Icon(Icons.description, size: 16),
            label: Text(S.get('view_detailed_report')),
          ),
          const SizedBox(width: 8),
          OutlinedButton.icon(
            onPressed: () => _regenerateReport(debateId),
            icon: const Icon(Icons.refresh, size: 16),
            label: Text(S.get('regenerate_report')),
          ),
        ]),
      ],
    );
  }

  Widget _buildStepCard({
    required String number,
    required String title,
    required String status,
    required Widget content,
  }) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header row: number + title + badge
          Row(children: [
            Text(number,
                style: TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: status == 'pending' ? const Color(0xFFE0E0E0) : Colors.black,
                )),
            const SizedBox(width: 12),
            Expanded(
                child: Text(title,
                    style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600))),
            _statusBadge(status),
          ]),
          const Divider(height: 16),
          content,
        ],
      ),
    );
  }

  Widget _statusBadge(String status) {
    Color bg, fg;
    String label;
    switch (status) {
      case 'completed':
        bg = const Color(0xFFE8F5E9);
        fg = const Color(0xFF2E7D32);
        label = 'COMPLETED';
      case 'processing':
        bg = const Color(0xFFFF5722);
        fg = Colors.white;
        label = 'PROCESSING';
      default:
        bg = const Color(0xFFF5F5F5);
        fg = const Color(0xFF999999);
        label = 'PENDING';
    }
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(4)),
      child: Text(label,
          style: TextStyle(color: fg, fontSize: 10, fontWeight: FontWeight.w600)),
    );
  }

  Widget _teamBadge(String? team) {
    final ds = ref.read(settingsProvider).debateSettings;
    final tA = ds['team_a_name'] as String? ?? 'Team A';
    final tB = ds['team_b_name'] as String? ?? 'Team B';
    final color = team == 'team_a'
        ? Colors.blue
        : team == 'team_b'
            ? Colors.red
            : Colors.amber;
    final label = team == 'team_a'
        ? S.teamName('team_a', teamAName: tA, teamBName: tB).toUpperCase()
        : team == 'team_b'
            ? S.teamName('team_b', teamAName: tA, teamBName: tB).toUpperCase()
            : 'JUDGE';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
          color: color.withOpacity(0.1),
          border: Border.all(color: color.withOpacity(0.3)),
          borderRadius: BorderRadius.circular(2)),
      child: Text(label,
          style: TextStyle(fontSize: 9, fontWeight: FontWeight.w600, color: color)),
    );
  }

  List<Widget> _buildAgentGroup(String groupLabel, List<AgentProfile> groupAgents) {
    if (groupAgents.isEmpty) return [];
    return [
      Padding(
        padding: const EdgeInsets.only(bottom: 4, top: 4),
        child: Text(groupLabel,
            style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: Color(0xFF999999))),
      ),
      ...groupAgents.map((agent) => _agentRow(agent)),
    ];
  }

  Widget _agentRow(AgentProfile agent) {
    final teamColor = agent.team == 'team_a'
        ? Colors.blue
        : agent.team == 'team_b'
            ? Colors.red
            : Colors.amber;
    final initial = agent.name.isNotEmpty ? agent.name[0] : '?';
    final debateState = ref.read(debateProvider);
    final canEdit = !['running', 'extended'].contains(debateState.status);

    return GestureDetector(
      onTap: () => _showAgentDetailDialog(agent, canEdit),
      child: MouseRegion(
        cursor: SystemMouseCursors.click,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 4),
          child: Row(children: [
            Container(
              width: 24,
              height: 24,
              decoration: BoxDecoration(color: teamColor, shape: BoxShape.circle),
              child: Center(
                  child: Text(initial,
                      style: const TextStyle(
                          color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold))),
            ),
            const SizedBox(width: 8),
            Expanded(
                child: Text(agent.name,
                    style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600))),
            Text(agent.specialty,
                style: const TextStyle(fontSize: 12, color: Color(0xFF666666))),
            const SizedBox(width: 4),
            Icon(Icons.info_outline, size: 14, color: Colors.grey.shade400),
          ]),
        ),
      ),
    );
  }

  void _showAgentDetailDialog(AgentProfile agent, bool canEdit) {
    final ds = ref.read(settingsProvider).debateSettings;
    final tA = ds['team_a_name'] as String? ?? 'Team A';
    final tB = ds['team_b_name'] as String? ?? 'Team B';
    final specialtyCtrl = TextEditingController(text: agent.specialty);
    final personalityCtrl = TextEditingController(text: agent.personality);
    final debateStyleCtrl = TextEditingController(text: agent.debateStyle);
    final backgroundCtrl = TextEditingController(text: agent.background);

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: Colors.white,
        title: Row(children: [
          Container(
            width: 32, height: 32,
            decoration: BoxDecoration(
              color: agent.team == 'team_a' ? Colors.blue
                  : agent.team == 'team_b' ? Colors.red : Colors.amber,
              shape: BoxShape.circle,
            ),
            child: Center(child: Text(
              agent.name.isNotEmpty ? agent.name[0] : '?',
              style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
            )),
          ),
          const SizedBox(width: 12),
          Expanded(child: Text(agent.name, style: const TextStyle(fontSize: 18))),
          if (agent.team != null)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
              decoration: BoxDecoration(
                color: agent.team == 'team_a' ? Colors.blue.withAlpha(25)
                    : agent.team == 'team_b' ? Colors.red.withAlpha(25) : Colors.amber.withAlpha(25),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                S.teamName(agent.team ?? '', teamAName: tA, teamBName: tB),
                style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600,
                  color: agent.team == 'team_a' ? Colors.blue : agent.team == 'team_b' ? Colors.red : Colors.amber),
              ),
            ),
        ]),
        content: SizedBox(
          width: 500,
          child: SingleChildScrollView(child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _agentField('전문분야', specialtyCtrl, canEdit),
              _agentField('성격', personalityCtrl, canEdit),
              _agentField('토론 스타일', debateStyleCtrl, canEdit),
              _agentField('배경', backgroundCtrl, canEdit, maxLines: 3),
              if (canEdit)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: DropdownButtonFormField<String>(
                    value: agent.llmOverride,
                    style: const TextStyle(fontSize: 13, color: Colors.black87),
                    decoration: const InputDecoration(
                      labelText: '모델',
                      labelStyle: TextStyle(fontSize: 11, color: Color(0xFF999999), fontWeight: FontWeight.w600),
                      isDense: true,
                      contentPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                      filled: true,
                      fillColor: Color(0xFFFAFAFA),
                    ),
                    items: [
                      const DropdownMenuItem(value: null, child: Text('Default')),
                      ...ref.read(settingsProvider).availableModels.map((m) {
                        final id = m['id'] as String? ?? '';
                        return DropdownMenuItem(
                          value: id,
                          child: Text(m['name'] as String? ?? id, overflow: TextOverflow.ellipsis),
                        );
                      }),
                    ],
                    onChanged: (val) {
                      ref.read(debateProvider.notifier).updateAgent(agent.agentId, {'llm_override': val});
                    },
                  ),
                )
              else if (agent.llmOverride != null && agent.llmOverride!.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Row(children: [
                    const Text('모델: ', style: TextStyle(fontSize: 12, color: Color(0xFF999999))),
                    Text(agent.llmOverride!, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
                  ]),
                ),
              if (!canEdit)
                const Padding(
                  padding: EdgeInsets.only(top: 12),
                  child: Text('토론 진행 중에는 수정할 수 없습니다. 일시정지 후 수정하세요.',
                      style: TextStyle(fontSize: 11, color: Color(0xFF999999), fontStyle: FontStyle.italic)),
                ),
            ],
          )),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('닫기')),
          if (canEdit)
            ElevatedButton(
              onPressed: () async {
                final updates = <String, dynamic>{};
                if (specialtyCtrl.text != agent.specialty) updates['specialty'] = specialtyCtrl.text;
                if (personalityCtrl.text != agent.personality) updates['personality'] = personalityCtrl.text;
                if (debateStyleCtrl.text != agent.debateStyle) updates['debate_style'] = debateStyleCtrl.text;
                if (backgroundCtrl.text != agent.background) updates['background'] = backgroundCtrl.text;
                if (updates.isNotEmpty) {
                  try {
                    await ref.read(debateProvider.notifier).updateAgent(agent.agentId, updates);
                    if (ctx.mounted) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text('에이전트 프로필이 수정되었습니다.')),
                      );
                    }
                  } catch (e) {
                    if (ctx.mounted) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text('수정 실패: $e')),
                      );
                    }
                  }
                }
                if (ctx.mounted) Navigator.pop(ctx);
              },
              child: const Text('저장'),
            ),
        ],
      ),
    );
  }

  Widget _agentField(String label, TextEditingController ctrl, bool canEdit, {int maxLines = 1}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(fontSize: 11, color: Color(0xFF999999), fontWeight: FontWeight.w600)),
          const SizedBox(height: 4),
          TextFormField(
            controller: ctrl,
            readOnly: !canEdit,
            maxLines: maxLines,
            style: const TextStyle(fontSize: 13),
            decoration: InputDecoration(
              isDense: true,
              contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
              filled: true,
              fillColor: canEdit ? const Color(0xFFFAFAFA) : const Color(0xFFF5F5F5),
            ),
          ),
        ],
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Step 03 content: debate progress
  // ---------------------------------------------------------------------------

  Widget _buildDebateProgressContent(List<DebateLogEntry> log, DebateState debateState) {
    // If log is empty and running, show phase indicator
    if (log.isEmpty && debateState.status == 'running') {
      return _buildPhaseIndicator(debateState);
    }

    if (log.isEmpty) {
      return const Text('토론 대기 중...',
          style: TextStyle(fontSize: 13, color: Color(0xFF999999)));
    }

    // Show debate log cards
    return Column(
      children: [
        ...log.asMap().entries.map((e) {
              // Assign GlobalKey for scroll navigation
              _logCardKeys.putIfAbsent(e.key, () => GlobalKey());
              // Match tracking is pre-computed in build()
              return Container(
                key: _logCardKeys[e.key],
                child: _buildLogCard(
                  e.value, debateState,
                  initiallyExpanded: (_cardExpanded[e.key] ?? false) ||
                      (e.value.entryType != 'judge_question' && e.value.entryType != 'qa_answer'),
                  cardIndex: e.key,
                ),
              );
            }),
        // Show phase indicator at the bottom if still running
        if (debateState.status == 'running')
          _buildPhaseIndicator(debateState),
      ],
    );
  }

  Widget _buildPhaseIndicator(DebateState debateState) {
    final ds = ref.read(settingsProvider).debateSettings;
    final tA = ds['team_a_name'] as String? ?? 'Team A';
    final tB = ds['team_b_name'] as String? ?? 'Team B';
    final phase = debateState.currentPhase ?? '';
    final team = debateState.currentTeam ?? '';
    final teamLabel = team == 'team_a'
        ? S.teamName('team_a', teamAName: tA, teamBName: tB)
        : team == 'team_b'
            ? S.teamName('team_b', teamAName: tA, teamBName: tB)
            : '';
    final phaseMsg = switch (phase) {
      'searching' => '$teamLabel 법률 검색 중...',
      'discussing' => '$teamLabel 내부 토론 중 (${debateState.discussionProgress}/${debateState.discussionTotal})',
      'statement' => '$teamLabel 대표 발언 중...',
      'judging' => '심판 평가 중...',
      _ => '토론 준비 중...',
    };

    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        children: [
          // Orange pulsing dot
          SizedBox(
            width: 24,
            height: 24,
            child: AnimatedBuilder(
              animation: _pulseAnimation,
              builder: (context, child) {
                final scale = 0.8 + (_pulseAnimation.value * 1.7);
                return Stack(
                  alignment: Alignment.center,
                  children: [
                    // Ripple ring
                    Container(
                      width: 10 * scale,
                      height: 10 * scale,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(
                            color: const Color(0xFFFF5722)
                                .withOpacity((2.5 / scale - 0.4).clamp(0.0, 1.0)),
                            width: 1),
                      ),
                    ),
                    // Center dot
                    Container(
                      width: 8,
                      height: 8,
                      decoration: const BoxDecoration(
                          color: Color(0xFFFF5722), shape: BoxShape.circle),
                    ),
                  ],
                );
              },
            ),
          ),
          const SizedBox(height: 12),
          Text(phaseMsg,
              style: const TextStyle(fontSize: 13, color: Color(0xFF666666))),
        ],
      ),
    );
  }

  /// Resolve agent_id (e.g. "team_a_1") to agent name (e.g. "김민준")
  String _resolveAgentName(String agentId, DebateState debateState) {
    for (final agent in debateState.agents) {
      if (agent.agentId == agentId) return agent.name;
    }
    return agentId;
  }

  /// Launch a legal URL using url_launcher.
  Future<void> _openLegalLink(String type, String detail) async {
    String url;
    // Clean detail: remove "대법원", "판결" etc. for cleaner URL
    final cleanDetail = detail.replaceAll(RegExp(r'(대법원|판결|선고)\s*'), '').trim();
    switch (type) {
      case 'court_precedent':
      case '판례':
        url = 'https://www.law.go.kr/precSc.do?tabMenuId=465&query=${Uri.encodeComponent(cleanDetail)}';
        break;
      case 'legal_statute':
      case '법령':
        // Clean parenthetical suffixes and article numbers for URL
        var cleanLaw = cleanDetail.replaceAll(RegExp(r'\s*[\(\[（【].*$'), '').trim();
        cleanLaw = cleanLaw.replaceAll(RegExp(r'\s+제\d+조.*$'), '').trim();
        url = 'https://www.law.go.kr/법령/${Uri.encodeComponent(cleanLaw)}';
        break;
      case 'constitutional_decision':
      case '헌재':
        url = 'https://www.law.go.kr/precSc.do?tabMenuId=465&query=${Uri.encodeComponent(cleanDetail)}';
        break;
      case '행심':
        url = 'https://www.law.go.kr/precSc.do?tabMenuId=465&query=${Uri.encodeComponent(cleanDetail)}';
        break;
      default:
        return;
    }
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  /// Resolve a citation type string to a canonical type key.
  String _canonicalCiteType(String raw) {
    switch (raw) {
      case '법령':
      case 'legal_statute':
        return 'legal_statute';
      case '판례':
      case 'court_precedent':
      case 'case_citation':
        return 'court_precedent';
      case '헌재':
      case 'constitutional_decision':
        return 'constitutional_decision';
      case '행심':
      case 'admin_tribunal':
        return 'admin_tribunal';
      case '문서':
      case 'uploaded_document':
        return 'uploaded_document';
      case 'legal_interpretation':
        return 'legal_interpretation';
      default:
        return raw;
    }
  }

  /// Get display label and color for a citation type.
  (String label, Color color, bool isClickable) _citeStyle(String canonType) {
    switch (canonType) {
      case 'legal_statute':
        return ('법령', const Color(0xFF4CAF50), true);
      case 'court_precedent':
        return ('판례', Colors.blue, true);
      case 'constitutional_decision':
        return ('헌재', Colors.purple, true);
      case '행심':
        return ('행심', Colors.teal, true);
      case 'uploaded_document':
        return ('문서', Colors.grey, false);
      default:
        return (canonType, Colors.blueGrey, false);
    }
  }

  /// Find a matching URL from evidence list by checking if evidence_id
  /// or source_detail contains the citation detail text.
  String? _findEvidenceUrl(String detail, List<dynamic> evidence) {
    for (final e in evidence) {
      if (e is! Map) continue;
      final evidenceId = (e['evidence_id'] ?? '').toString();
      final sourceDetail = (e['source_detail'] ?? '').toString();
      final url = (e['url'] ?? '').toString();
      if (url.isNotEmpty &&
          (evidenceId.contains(detail) ||
           sourceDetail.contains(detail) ||
           detail.contains(evidenceId))) {
        return url;
      }
    }
    return null;
  }

  /// Build a rich-text widget from a statement string, parsing citation
  /// patterns into clickable links. Supports both new Korean format
  /// ([법령: ...], [판례: ...], [헌재: ...], [행심: ...], [문서: ...])
  /// and old [CITE:type:id] format.
  /// Split text into TextSpans, highlighting matches for _searchQuery.
  /// Precompute the global start index for every text block in the debate.
  void _precomputeBlockOffsets(List<DebateLogEntry> log, List<Verdict> verdicts) {
    _blockStartIndex.clear();
    if (_searchQuery.isEmpty) return;

    int cumulative = 0;
    for (int i = 0; i < log.length; i++) {
      // Main statement (speaker + statement, matching _searchMatches logic)
      _blockStartIndex['card_${i}_main'] = cumulative;
      cumulative += _countMatchesInText('${log[i].speaker} ${log[i].statement}');

      // Evidence chips excluded — not highlighted in rendering, so skip offset tracking.

      // Internal discussion entries (shared text extraction)
      for (int di = 0; di < log[i].internalDiscussion.length; di++) {
        _blockStartIndex['card_${i}_disc_$di'] = cumulative;
        final content = _discussionSearchText(log[i].internalDiscussion[di]);
        cumulative += _countMatchesInText(content);
      }
    }

    // Verdict sections
    for (int vi = 0; vi < verdicts.length; vi++) {
      _blockStartIndex['verdict_$vi'] = cumulative;
      cumulative += _countMatchesInText(verdicts[vi].reasoning);
    }
  }

  /// Shared text extraction for discussion entries (used by both _searchMatches and _precomputeBlockOffsets).
  String _discussionSearchText(dynamic d) {
    if (d is Map) return d['content']?.toString() ?? '';
    return d.toString();
  }

  /// Count search matches in text without rendering (for skipping collapsed sections).
  int _countMatchesInText(String text) {
    if (_searchQuery.isEmpty) return 0;
    final q = _searchQuery.toLowerCase();
    final lower = text.toLowerCase();
    int count = 0, idx = 0;
    while ((idx = lower.indexOf(q, idx)) != -1) {
      count++;
      idx += q.length;
    }
    return count;
  }

  /// Highlight search matches using precomputed `startIndex`.
  /// [startIndex] is the global index of the first match in this text block.
  List<InlineSpan> _highlightText(String text, {int startIndex = 0}) {
    if (_searchQuery.isEmpty) {
      return [TextSpan(text: text)];
    }
    final query = _searchQuery.toLowerCase();
    final spans = <InlineSpan>[];
    int start = 0;
    int localCount = 0;
    final lowerText = text.toLowerCase();

    while (true) {
      final idx = lowerText.indexOf(query, start);
      if (idx == -1) {
        if (start < text.length) {
          spans.add(TextSpan(text: text.substring(start)));
        }
        break;
      }
      if (idx > start) {
        spans.add(TextSpan(text: text.substring(start, idx)));
      }
      final globalIdx = startIndex + localCount;
      final isCurrent = globalIdx == _currentMatchIndex;
      if (isCurrent) _currentMatchKey = GlobalKey();
      spans.add(WidgetSpan(
        alignment: PlaceholderAlignment.baseline,
        baseline: TextBaseline.alphabetic,
        child: Container(
          key: isCurrent ? _currentMatchKey : null,
          color: isCurrent ? const Color(0xFFFFAB00) : const Color(0xFFFFF9C4),
          child: Text(
            text.substring(idx, idx + query.length),
            style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
          ),
        ),
      ));
      localCount++;
      start = idx + query.length;
    }
    return spans;
  }

  // (old convenience _highlightText removed — using unified version above)

  /// Renders chip text with search highlighting support.
  Widget _chipSearchable(String text, TextStyle baseStyle, {int startIndex = 0}) {
    if (_searchQuery.isEmpty || !text.toLowerCase().contains(_searchQuery.toLowerCase())) {
      return Text(text, style: baseStyle);
    }
    final query = _searchQuery.toLowerCase();
    final idx = text.toLowerCase().indexOf(query);
    final isCurrent = startIndex == _currentMatchIndex;
    return RichText(
      text: TextSpan(
        style: baseStyle,
        children: [
          if (idx > 0) TextSpan(text: text.substring(0, idx)),
          TextSpan(
            text: text.substring(idx, idx + query.length),
            style: baseStyle.copyWith(
              backgroundColor: isCurrent ? const Color(0xFFFFAB00) : const Color(0xFFFFF9C4),
            ),
          ),
          if (idx + query.length < text.length)
            TextSpan(text: text.substring(idx + query.length)),
        ],
      ),
    );
  }

  Widget _buildRichStatement(String text, List<dynamic> evidence, {int startIndex = 0}) {
    // Combined regex: Korean format  OR  old CITE format  OR  unverified warning
    final regex = RegExp(
      r'\[(법령|판례|헌재|행심|문서|case_citation|legal_statute|court_precedent|constitutional_decision|admin_tribunal|legal_interpretation):\s*([^\]]+)\]'
      r'|'
      r'\[CITE:([^:]+):([^\]]+)\]'
      r'|'
      r'\[⚠ 미확인 인용:\s*([^\]]+)\]',
    );
    final spans = <InlineSpan>[];
    int lastEnd = 0;
    int runningOffset = startIndex;

    for (final match in regex.allMatches(text)) {
      // Add text before this match (with search highlighting)
      if (match.start > lastEnd) {
        final segment = text.substring(lastEnd, match.start);
        spans.addAll(_highlightText(segment, startIndex: runningOffset));
        runningOffset += _countMatchesInText(segment);
      }

      // Determine which format matched
      final String rawType;
      final String detail;
      if (match.group(5) != null) {
        // Unverified citation warning: [⚠ 미확인 인용: ...]
        // (text before match already handled above)
        spans.add(WidgetSpan(
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
            margin: const EdgeInsets.symmetric(horizontal: 2),
            decoration: BoxDecoration(
              color: Colors.red.withAlpha(25),
              borderRadius: BorderRadius.circular(3),
              border: Border.all(color: Colors.red.withAlpha(77)),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.warning_amber, size: 10, color: Colors.red),
                const SizedBox(width: 2),
                _chipSearchable(
                  '⚠ ${match.group(5)!.trim()}',
                  const TextStyle(fontSize: 11, color: Colors.red, fontWeight: FontWeight.w600),
                ),
              ],
            ),
          ),
        ));
        lastEnd = match.end;
        continue;
      } else if (match.group(1) != null) {
        // Korean format: [법령: ...]
        rawType = match.group(1)!;
        detail = match.group(2)!.trim();
      } else {
        // Old format: [CITE:type:detail]
        rawType = match.group(3)!;
        detail = match.group(4)!.trim();
      }

      final canonType = _canonicalCiteType(rawType);
      final (label, color, isClickable) = _citeStyle(canonType);

      if (isClickable) {
        // Check if evidence has a direct URL
        final evidenceUrl = _findEvidenceUrl(detail, evidence);

        spans.add(WidgetSpan(
          child: GestureDetector(
            onTap: () {
              if (evidenceUrl != null) {
                final uri = Uri.parse(evidenceUrl);
                launchUrl(uri, mode: LaunchMode.externalApplication);
              } else {
                _openLegalLink(canonType, detail);
              }
            },
            child: MouseRegion(
              cursor: SystemMouseCursors.click,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                margin: const EdgeInsets.symmetric(horizontal: 2),
                decoration: BoxDecoration(
                  color: color.withAlpha(25),
                  borderRadius: BorderRadius.circular(3),
                  border: Border.all(color: color.withAlpha(77)),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.open_in_new, size: 10, color: color),
                    const SizedBox(width: 2),
                    _chipSearchable('[$label: $detail]', TextStyle(fontSize: 11, color: color, fontWeight: FontWeight.w600)),
                  ],
                ),
              ),
            ),
          ),
        ));
      } else {
        // Non-clickable chip (e.g. 문서)
        spans.add(WidgetSpan(
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
            margin: const EdgeInsets.symmetric(horizontal: 2),
            decoration: BoxDecoration(
              color: color.withAlpha(25),
              borderRadius: BorderRadius.circular(3),
              border: Border.all(color: color.withAlpha(77)),
            ),
            child: _chipSearchable('[$label: $detail]', TextStyle(fontSize: 11, color: color, fontWeight: FontWeight.w600)),
          ),
        ));
      }

      lastEnd = match.end;
    }

    // Add remaining text (with search highlighting)
    if (lastEnd < text.length) {
      spans.addAll(_highlightText(text.substring(lastEnd), startIndex: runningOffset));
    }

    if (spans.isEmpty) {
      return SelectableText(text, style: const TextStyle(fontSize: 13, color: Colors.black87, height: 1.6));
    }

    return SelectableText.rich(
      TextSpan(
        style: const TextStyle(fontSize: 13, color: Colors.black87, height: 1.6),
        children: spans,
      ),
    );
  }

  Widget _buildLogCard(DebateLogEntry entry, DebateState debateState, {bool initiallyExpanded = false, int cardIndex = -1}) {
    final teamColor = entry.team == 'team_a'
        ? Colors.blue
        : entry.team == 'team_b'
            ? Colors.red
            : Colors.amber;
    final agentName = _resolveAgentName(entry.speaker, debateState);
    final initial = agentName.isNotEmpty ? agentName[0] : '?';

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border.all(color: const Color(0xFFEAEAEA)),
        borderRadius: BorderRadius.circular(2),
      ),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          controller: cardIndex >= 0
              ? (_cardControllers.putIfAbsent(cardIndex, () => ExpansionTileController()))
              : null,
          initiallyExpanded: initiallyExpanded,
          tilePadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
          childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          leading: Container(
            width: 24,
            height: 24,
            decoration: BoxDecoration(color: teamColor, shape: BoxShape.circle),
            child: Center(
                child: Text(initial,
                    style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.bold))),
          ),
          title: Row(children: [
            Text(agentName,
                style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
            const Spacer(),
            _teamBadge(entry.team),
            const SizedBox(width: 6),
            Text('R${entry.round}',
                style: const TextStyle(
                    fontSize: 10,
                    color: Color(0xFFBBBBBB),
                    fontFamily: 'monospace')),
          ]),
          subtitle: Text(
            entry.statement.length > 80
                ? '${entry.statement.substring(0, 80)}...'
                : entry.statement,
            style: const TextStyle(fontSize: 11, color: Color(0xFF999999)),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
          children: [
            const Divider(height: 1, color: Color(0xFFF5F5F5)),
            const SizedBox(height: 12),
            // Full statement text (selectable)
            SelectionArea(
              child: _buildRichStatement(entry.statement, entry.evidence,
                  startIndex: (_blockStartIndex['card_${cardIndex}_main'] ?? 0) +
                      _countMatchesInText(entry.speaker)),
            ),
            // Evidence chips
            if (entry.evidence.isNotEmpty) ...[
              const SizedBox(height: 12),
              Wrap(
                  spacing: 6,
                  runSpacing: 4,
                  children: entry.evidence.map((e) {
                    final rawId = (e is Map ? e['evidence_id'] ?? '' : '').toString();
                    final sourceDetail = (e is Map ? e['source_detail'] ?? '' : '').toString();
                    final sourceType = (e is Map ? e['source_type'] ?? '' : '').toString();
                    final evidenceUrl = (e is Map ? e['url'] ?? '' : '').toString();
                    // sourceType → Korean prefix mapping
                    final typePrefix = sourceType.contains('precedent') ? '판례'
                        : sourceType.contains('statute') ? '법령'
                        : sourceType.contains('constitutional') ? '헌재'
                        : sourceType.contains('admin_tribunal') ? '행심'
                        : sourceType.contains('document') || sourceType.contains('uploaded') ? '문서'
                        : '';
                    // Display: prefer source_detail (법령명/사건번호), then evidence_id, then source_type
                    final displayLabel = sourceDetail.isNotEmpty ? sourceDetail
                        : rawId.isNotEmpty && !rawId.contains('-') ? rawId  // not UUID
                        : sourceType.isNotEmpty ? sourceType
                        : rawId;
                    final evidenceId = rawId.isNotEmpty ? rawId : displayLabel;
                    // Chip display text: prepend type prefix if available
                    final chipDisplayText = typePrefix.isNotEmpty && evidenceId.isNotEmpty
                        ? '$typePrefix: $evidenceId'
                        : displayLabel.isNotEmpty ? displayLabel : evidenceId;
                    // Color coding by source type
                    final Color chipBg;
                    final Color chipBorder;
                    final Color chipText;
                    final bool isClickable;
                    if (sourceType.contains('statute')) {
                      chipBg = const Color(0xFFE8F5E9);
                      chipBorder = const Color(0xFF4CAF50);
                      chipText = const Color(0xFF2E7D32);
                      isClickable = true;
                    } else if (sourceType.contains('precedent')) {
                      chipBg = const Color(0xFFE3F2FD);
                      chipBorder = Colors.blue;
                      chipText = Colors.blue;
                      isClickable = true;
                    } else if (sourceType.contains('constitutional')) {
                      chipBg = const Color(0xFFF3E5F5);
                      chipBorder = Colors.purple;
                      chipText = Colors.purple;
                      isClickable = true;
                    } else {
                      chipBg = const Color(0xFFF0F0F0);
                      chipBorder = const Color(0xFFE0E0E0);
                      chipText = const Color(0xFF666666);
                      isClickable = false;
                    }
                    return GestureDetector(
                      onTap: () {
                        if (evidenceUrl.isNotEmpty) {
                          launchUrl(Uri.parse(evidenceUrl), mode: LaunchMode.externalApplication);
                        } else if (sourceType.contains('precedent')) {
                          _openLegalLink('court_precedent', evidenceId);
                        } else if (sourceType.contains('statute')) {
                          _openLegalLink('legal_statute', evidenceId);
                        } else if (sourceType.contains('constitutional')) {
                          _openLegalLink('constitutional_decision', evidenceId);
                        }
                      },
                      child: MouseRegion(
                        cursor: isClickable || evidenceUrl.isNotEmpty
                            ? SystemMouseCursors.click
                            : SystemMouseCursors.basic,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(
                            color: chipBg,
                            border: Border.all(color: chipBorder.withAlpha(77)),
                            borderRadius: BorderRadius.circular(2),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              if (isClickable || evidenceUrl.isNotEmpty)
                                Padding(
                                  padding: const EdgeInsets.only(right: 4),
                                  child: Icon(Icons.open_in_new, size: 10, color: chipText),
                                ),
                              Text(
                                chipDisplayText.length > 60 ? '${chipDisplayText.substring(0, 60)}...' : chipDisplayText,
                                style: TextStyle(fontSize: 11, color: chipText, fontWeight: FontWeight.w500),
                              ),
                            ],
                          ),
                        ),
                      ),
                    );
                  }).toList()),
            ],
            // Internal discussion (collapsed by default)
            if (entry.internalDiscussion.isNotEmpty) ...[
              const SizedBox(height: 8),
              GestureDetector(
                onTap: () => setState(() {
                  _discussionExpanded[cardIndex] = !(_discussionExpanded[cardIndex] ?? false);
                }),
                child: MouseRegion(
                  cursor: SystemMouseCursors.click,
                  child: Row(children: [
                    Icon(
                      (_discussionExpanded[cardIndex] ?? false) ? Icons.expand_less : Icons.expand_more,
                      size: 18, color: const Color(0xFF666666),
                    ),
                    const SizedBox(width: 4),
                    Text(
                      '내부 토론 (${entry.internalDiscussion.length}건)',
                      style: const TextStyle(fontSize: 12, color: Color(0xFF666666), fontWeight: FontWeight.w500),
                    ),
                  ]),
                ),
              ),
              if (_discussionExpanded[cardIndex] ?? false)
                AnimatedSize(
                  duration: const Duration(milliseconds: 200),
                  child: Padding(
                    padding: const EdgeInsets.only(top: 6),
                    child: Column(
                      children: entry.internalDiscussion.asMap().entries.map<Widget>((de) {
                        final di = de.key;
                        final d = de.value;
                        final speaker = d['speaker'] as String? ?? '?';
                        final content = d['content'] as String? ?? '';
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 6),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              CircleAvatar(
                                radius: 10,
                                backgroundColor: const Color(0xFFE5E5E5),
                                child: Text(speaker.isNotEmpty ? speaker[0] : '?',
                                    style: const TextStyle(fontSize: 10, color: Colors.black54)),
                              ),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(speaker,
                                        style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600)),
                                    _buildRichStatement(content, entry.evidence,
                                        startIndex: _blockStartIndex['card_${cardIndex}_disc_$di'] ?? 0),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        );
                      }).toList(),
                    ),
                  ),
                ),
              // (No Builder side-effect needed — using precomputed _blockStartIndex)
            ],
            // Timestamp + elapsed + tokens footer
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerRight,
              child: Text(
                _buildLogFooterText(entry),
                style: const TextStyle(
                    fontSize: 10, color: Color(0xFFBBBBBB), fontFamily: 'monospace'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _buildLogFooterText(DebateLogEntry entry) {
    final hh = entry.timestamp.hour.toString().padLeft(2, '0');
    final mm = entry.timestamp.minute.toString().padLeft(2, '0');
    var text = '$hh:$mm';
    if (entry.elapsedSeconds != null) {
      final m = entry.elapsedSeconds! ~/ 60;
      final s = entry.elapsedSeconds! % 60;
      text += ' \u00b7 ${m}m ${s}s';
    }
    if (entry.tokenUsage != null) {
      final total = entry.tokenUsage!['total'] ?? 0;
      text += ' \u00b7 ${_formatTokenCount(total)} tok';
    }
    return text;
  }

  String _formatTokenCount(dynamic count) {
    final n = (count is int) ? count : (count as num).toInt();
    if (n >= 1000) {
      return '${(n / 1000).toStringAsFixed(1)}k';
    }
    return n.toString();
  }

  // ---------------------------------------------------------------------------
  // Intervention panel
  // ---------------------------------------------------------------------------

  Widget _buildInterventionPanel(DebateState debateState) {
    final ds = ref.read(settingsProvider).debateSettings;
    final tA = ds['team_a_name'] as String? ?? 'Team A';
    final tB = ds['team_b_name'] as String? ?? 'Team B';
    final isActive = debateState.status == 'running';

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        // Recent interventions (above input)
        if (_interventions.isNotEmpty)
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
            decoration: const BoxDecoration(
              color: Color(0xFFF5F5F0),
              border: Border(bottom: BorderSide(color: Color(0xFFE5E5E5))),
            ),
            constraints: const BoxConstraints(maxHeight: 100),
            child: ListView(
              shrinkWrap: true,
              children: _interventions.reversed.take(3).map((iv) {
                const icon = '💡';
                final content = (iv['content'] as String? ?? '');
                final round = iv['round'] ?? 0;
                final displayContent = content.length > 60
                    ? '${content.substring(0, 60)}...'
                    : content;
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2),
                  child: Text(
                    '$icon R$round: $displayContent',
                    style: const TextStyle(fontSize: 11, color: Color(0xFF666666)),
                  ),
                );
              }).toList(),
            ),
          ),
        // Input row
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          color: const Color(0xFFFAFAFA),
          child: Row(
            children: [
              // Team radio buttons.
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Radio<String>(
                    value: 'team_a',
                    groupValue: _interventionTeam,
                    onChanged: isActive
                        ? (val) => setState(() => _interventionTeam = val!)
                        : null,
                  ),
                  Text(S.teamName('team_a', teamAName: tA, teamBName: tB),
                      style: const TextStyle(fontSize: 13, color: Colors.black87)),
                  const SizedBox(width: 4),
                  Radio<String>(
                    value: 'team_b',
                    groupValue: _interventionTeam,
                    onChanged: isActive
                        ? (val) => setState(() => _interventionTeam = val!)
                        : null,
                  ),
                  Text(S.teamName('team_b', teamAName: tA, teamBName: tB),
                      style: const TextStyle(fontSize: 13, color: Colors.black87)),
                  const SizedBox(width: 4),
                  Radio<String>(
                    value: 'both',
                    groupValue: _interventionTeam,
                    onChanged: isActive
                        ? (val) => setState(() => _interventionTeam = val!)
                        : null,
                  ),
                  Text(S.get('both_teams'),
                      style: const TextStyle(fontSize: 13, color: Colors.black87)),
                ],
              ),
              const SizedBox(width: 8),
              // Hint text field.
              Expanded(
                child: TextField(
                  controller: _interventionCtrl,
                  enabled: isActive,
                  style: const TextStyle(color: Colors.black87),
                  decoration: InputDecoration(
                    hintText: S.get('enter_hint'),
                    hintStyle: const TextStyle(color: Color(0xFF999999)),
                    isDense: true,
                    contentPadding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  ),
                  onSubmitted: (_) => _sendIntervention(),
                ),
              ),
              const SizedBox(width: 8),
              // Upload evidence button.
              IconButton(
                icon: const Icon(Icons.attach_file),
                tooltip: S.get('upload_evidence'),
                color: Colors.black87,
                onPressed: isActive ? _uploadEvidence : null,
              ),
              const SizedBox(width: 4),
              // Send button.
              FilledButton(
                onPressed: isActive ? _sendIntervention : null,
                child: Text(S.get('send')),
              ),
            ],
          ),
        ),
      ],
    );
  }

  // ---------------------------------------------------------------------------
  // Paused overlay
  // ---------------------------------------------------------------------------

  // _buildPausedOverlay removed — pause/resume is now a toggle button,
  // model changes happen in agent detail dialog.
}

// Intent classes for Ctrl+F search
class _SearchMatch {
  final int cardIndex;
  final bool isInToggle;
  final String toggleType;  // 'discussion', 'timeline', 'key_facts', ''

  const _SearchMatch({
    required this.cardIndex,
    this.isInToggle = false,
    this.toggleType = '',
  });
}

class _OpenSearchIntent extends Intent {
  const _OpenSearchIntent();
}

class _CloseSearchIntent extends Intent {
  const _CloseSearchIntent();
}


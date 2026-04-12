import 'package:flutter/material.dart';

import '../models/debate.dart';

/// Scrollable timeline of debate log entries displayed as styled cards.
///
/// Each entry shows the speaker, round, team color, statement text
/// (expandable if long), evidence chips, and a relative timestamp.
class DebateTimelineWidget extends StatelessWidget {
  /// Ordered list of debate log entries to display.
  final List<DebateLogEntry> entries;

  /// Optional scroll controller for programmatic scrolling.
  final ScrollController? scrollController;

  const DebateTimelineWidget({
    super.key,
    required this.entries,
    this.scrollController,
  });

  @override
  Widget build(BuildContext context) {
    if (entries.isEmpty) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.forum_outlined, size: 48, color: Color(0xFFCCCCCC)),
            SizedBox(height: 12),
            Text(
              'Waiting for debate to start...',
              style: TextStyle(color: Color(0xFF999999), fontSize: 14),
            ),
          ],
        ),
      );
    }

    return ListView.builder(
      controller: scrollController,
      padding: const EdgeInsets.all(8),
      itemCount: entries.length,
      itemBuilder: (context, index) {
        return _TimelineEntryCard(
          entry: entries[index],
          isFirst: index == 0,
          isLast: index == entries.length - 1,
        );
      },
    );
  }
}

/// A single card in the debate timeline.
class _TimelineEntryCard extends StatefulWidget {
  final DebateLogEntry entry;
  final bool isFirst;
  final bool isLast;

  const _TimelineEntryCard({
    required this.entry,
    required this.isFirst,
    required this.isLast,
  });

  @override
  State<_TimelineEntryCard> createState() => _TimelineEntryCardState();
}

class _TimelineEntryCardState extends State<_TimelineEntryCard> {
  bool _isExpanded = false;

  /// Maximum number of characters to show before truncating.
  static const int _truncateLength = 200;

  /// Determine team color for visual indicators.
  Color get _teamColor {
    switch (widget.entry.team) {
      case 'team_a':
        return Colors.blue;
      case 'team_b':
        return Colors.red;
      case 'judge':
        return Colors.amber;
      default:
        return Colors.grey;
    }
  }

  /// Human-readable team label.
  String get _teamLabel {
    switch (widget.entry.team) {
      case 'team_a':
        return 'Team A';
      case 'team_b':
        return 'Team B';
      case 'judge':
        return 'Judge';
      default:
        return widget.entry.team;
    }
  }

  /// Format timestamp as relative time (e.g. "2 min ago").
  String _relativeTime(DateTime dt) {
    final now = DateTime.now();
    final diff = now.difference(dt);

    if (diff.inSeconds < 60) return 'just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes} min ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    return '${diff.inDays}d ago';
  }

  @override
  Widget build(BuildContext context) {
    final entry = widget.entry;
    final statement = entry.statement;
    final isLong = statement.length > _truncateLength;
    final displayText = (!_isExpanded && isLong)
        ? '${statement.substring(0, _truncateLength)}...'
        : statement;

    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: IntrinsicHeight(
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Left color bar indicating team.
            Container(
              width: 4,
              decoration: BoxDecoration(
                color: _teamColor,
                borderRadius: BorderRadius.vertical(
                  top: widget.isFirst
                      ? const Radius.circular(4)
                      : Radius.zero,
                  bottom: widget.isLast
                      ? const Radius.circular(4)
                      : Radius.zero,
                ),
              ),
            ),
            const SizedBox(width: 8),
            // Card content.
            Expanded(
              child: Card(
                margin: EdgeInsets.zero,
                elevation: 1,
                color: Colors.white,
                surfaceTintColor: Colors.transparent,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                  side: const BorderSide(color: Color(0xFFE5E5E5)),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Header: avatar, name, round badge.
                      _buildHeader(entry),
                      const SizedBox(height: 8),

                      // Statement text.
                      Text(
                        displayText,
                        style: const TextStyle(fontSize: 13, height: 1.5, color: Colors.black87),
                      ),

                      // Expand / collapse toggle for long statements.
                      if (isLong) ...[
                        const SizedBox(height: 4),
                        GestureDetector(
                          onTap: () =>
                              setState(() => _isExpanded = !_isExpanded),
                          child: Text(
                            _isExpanded ? 'Show less' : 'Show more',
                            style: TextStyle(
                              fontSize: 12,
                              color: Theme.of(context).colorScheme.primary,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ),
                      ],

                      // Evidence chips row.
                      if (entry.evidence.isNotEmpty) ...[
                        const SizedBox(height: 8),
                        _buildEvidenceChips(entry.evidence),
                      ],

                      // Timestamp.
                      const SizedBox(height: 6),
                      Text(
                        _relativeTime(entry.timestamp),
                        style:
                            const TextStyle(fontSize: 10, color: Color(0xFFCCCCCC)),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Build the header row with avatar, name, and round badge.
  Widget _buildHeader(DebateLogEntry entry) {
    return Row(
      children: [
        // Agent avatar circle with first letter.
        CircleAvatar(
          radius: 14,
          backgroundColor: _teamColor.withValues(alpha: 0.2),
          child: Text(
            entry.speaker.isNotEmpty ? entry.speaker[0].toUpperCase() : '?',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.bold,
              color: _teamColor,
            ),
          ),
        ),
        const SizedBox(width: 8),
        // Agent name.
        Expanded(
          child: Text(
            entry.speaker,
            style: TextStyle(
              fontWeight: FontWeight.w600,
              color: Color.lerp(_teamColor, Colors.black, 0.3),
            ),
            overflow: TextOverflow.ellipsis,
          ),
        ),
        // Round badge.
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
          decoration: BoxDecoration(
            color: _teamColor.withValues(alpha: 0.15),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: _teamColor.withValues(alpha: 0.3),
            ),
          ),
          child: Text(
            'R${entry.round} | $_teamLabel',
            style: TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w500,
              color: _teamColor.withValues(alpha: 0.9),
            ),
          ),
        ),
      ],
    );
  }

  /// Build the horizontal row of evidence chips.
  Widget _buildEvidenceChips(List<dynamic> evidence) {
    return Wrap(
      spacing: 4,
      runSpacing: 4,
      children: evidence.map((e) {
        final label = _evidenceLabel(e);
        final icon = _evidenceIcon(e);
        return Chip(
          avatar: Text(icon, style: const TextStyle(fontSize: 12)),
          label: Text(
            label.length > 35 ? '${label.substring(0, 35)}...' : label,
            style: const TextStyle(fontSize: 10, color: Colors.black87),
          ),
          materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
          visualDensity: VisualDensity.compact,
          padding: EdgeInsets.zero,
          backgroundColor: Colors.white,
          side: const BorderSide(color: Color(0xFFE5E5E5)),
        );
      }).toList(),
    );
  }

  /// Extract a display label from an evidence entry.
  String _evidenceLabel(dynamic e) {
    if (e is Map) {
      return (e['source_detail'] ?? e['content'] ?? 'Evidence').toString();
    }
    return e.toString();
  }

  /// Return an icon string based on evidence source type.
  String _evidenceIcon(dynamic e) {
    if (e is Map) {
      switch (e['source_type']) {
        case 'statute':
          return '\u{1F4DC}'; // scroll emoji
        case 'precedent':
          return '\u{2696}'; // scales emoji
        case 'document':
          return '\u{1F4C4}'; // document emoji
        case 'graph':
          return '\u{1F517}'; // link emoji
        default:
          return '\u{1F4CE}'; // paperclip
      }
    }
    return '\u{1F4CE}';
  }
}

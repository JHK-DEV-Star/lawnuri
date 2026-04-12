import 'package:flutter/material.dart';

import '../l10n/app_strings.dart';

/// Compact card displaying a single piece of evidence with source type icon,
/// source detail, relevance snippet, and team color indicator.
class EvidenceCardWidget extends StatelessWidget {
  /// Evidence data map.
  ///
  /// Expected keys: "source_type", "source_detail", "content",
  /// "submitted_by" (or "team"), and optionally "relevance".
  final Map<String, dynamic> evidence;

  const EvidenceCardWidget({
    super.key,
    required this.evidence,
  });

  /// Determine team color from the evidence data.
  Color get _teamColor {
    final team =
        evidence['team'] as String? ?? evidence['submitted_by'] as String? ?? '';
    if (team.contains('team_a')) return Colors.blue;
    if (team.contains('team_b')) return Colors.red;
    return Colors.grey;
  }

  /// Return an icon string for the evidence source type.
  String get _sourceIcon {
    switch (evidence['source_type'] as String? ?? '') {
      case 'statute':
        return '\u{1F4DC}'; // scroll
      case 'precedent':
        return '\u{2696}\u{FE0F}'; // scales of justice
      case 'document':
        return '\u{1F4C4}'; // document
      case 'graph':
        return '\u{1F517}'; // link
      default:
        return '\u{1F4CE}'; // paperclip
    }
  }

  /// Return a human-readable label for the source type.
  String get _sourceTypeLabel {
    switch (evidence['source_type'] as String? ?? '') {
      case 'statute':
        return S.get('statute');
      case 'precedent':
        return S.get('precedent');
      case 'document':
        return S.get('document');
      case 'graph':
        return S.get('knowledge_graph');
      default:
        return S.get('evidence_label');
    }
  }

  @override
  Widget build(BuildContext context) {
    final sourceDetail = evidence['source_detail'] as String? ?? '';
    final content = evidence['content'] as String? ?? '';
    final relevance = evidence['relevance'] as String? ?? '';

    return Card(
      margin: const EdgeInsets.only(bottom: 6),
      clipBehavior: Clip.antiAlias,
      color: Colors.white,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: const BorderSide(color: Color(0xFFE5E5E5)),
      ),
      child: IntrinsicHeight(
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(width: 4, color: _teamColor),

            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(10),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text(
                          _sourceIcon,
                          style: const TextStyle(fontSize: 16),
                        ),
                        const SizedBox(width: 6),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: _teamColor.withValues(alpha: 0.1),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            _sourceTypeLabel,
                            style: TextStyle(
                              fontSize: 10,
                              fontWeight: FontWeight.w600,
                              color: _teamColor,
                            ),
                          ),
                        ),
                        const Spacer(),
                        // Submitted by label.
                        if (evidence['speaker'] != null)
                          Text(
                            evidence['speaker'] as String,
                            style: const TextStyle(
                              fontSize: 10,
                              color: Color(0xFF999999),
                            ),
                          ),
                      ],
                    ),

                    // Source detail text.
                    if (sourceDetail.isNotEmpty) ...[
                      const SizedBox(height: 6),
                      Text(
                        sourceDetail,
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w500,
                          color: Colors.black87,
                        ),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],

                    // Content / relevance snippet.
                    if (content.isNotEmpty || relevance.isNotEmpty) ...[
                      const SizedBox(height: 4),
                      Text(
                        relevance.isNotEmpty ? relevance : content,
                        style: const TextStyle(
                          fontSize: 11,
                          color: Color(0xFF666666),
                          height: 1.3,
                        ),
                        maxLines: 3,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

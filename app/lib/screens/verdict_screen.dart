import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../providers/debate_provider.dart';
import '../models/verdict.dart';
import '../l10n/app_strings.dart';

/// Screen displaying judge verdict results after a debate completes.
/// MiroFish light theme with numbered step-card sections.
class VerdictScreen extends ConsumerStatefulWidget {
  final String debateId;
  const VerdictScreen({super.key, required this.debateId});

  @override
  ConsumerState<VerdictScreen> createState() => _VerdictScreenState();
}

class _VerdictScreenState extends ConsumerState<VerdictScreen> {
  static const _borderColor = Color(0xFFE5E5E5);

  @override
  void initState() {
    super.initState();
    Future.microtask(() {
      final state = ref.read(debateProvider);
      // Load the debate if not already loaded for this ID.
      if (state.debateId != widget.debateId || state.verdicts.isEmpty) {
        ref.read(debateProvider.notifier).loadDebate(widget.debateId);
      }
    });
  }

  /// Extend the debate by 5 additional rounds and navigate back.
  Future<void> _continueDebate() async {
    final notifier = ref.read(debateProvider.notifier);
    await notifier.extendDebate(additionalRounds: 5);
    await notifier.resumeDebate();
    if (mounted) {
      context.go('/debate/${widget.debateId}');
    }
  }

  /// Compute the overall result string from the list of verdicts.
  String _computeResult(List<Verdict> verdicts) {
    if (verdicts.isEmpty) return 'No verdicts available';

    int teamACount = 0;
    int teamBCount = 0;
    int equalCount = 0;

    for (final v in verdicts) {
      switch (v.verdict) {
        case 'team_a':
          teamACount++;
          break;
        case 'team_b':
          teamBCount++;
          break;
        default:
          equalCount++;
      }
    }

    if (teamACount > teamBCount) {
      return 'Team A Superior - $teamACount/${verdicts.length} Judges';
    } else if (teamBCount > teamACount) {
      return 'Team B Superior - $teamBCount/${verdicts.length} Judges';
    } else {
      return 'Equal / Split - $equalCount/${verdicts.length} Judges';
    }
  }

  /// Get the dominant verdict color.
  Color _resultColor(List<Verdict> verdicts) {
    if (verdicts.isEmpty) return Colors.grey;

    int teamACount = 0;
    int teamBCount = 0;

    for (final v in verdicts) {
      if (v.verdict == 'team_a') teamACount++;
      if (v.verdict == 'team_b') teamBCount++;
    }

    if (teamACount > teamBCount) return Colors.blue;
    if (teamBCount > teamACount) return Colors.red;
    return Colors.amber;
  }

  /// Get a color for a single verdict value.
  Color _verdictColor(String verdict) {
    switch (verdict) {
      case 'team_a':
        return Colors.blue;
      case 'team_b':
        return Colors.red;
      case 'draw':
        return Colors.amber;
      default:
        return Colors.grey;
    }
  }

  /// Get a human-readable label for a verdict value.
  String _verdictLabel(String verdict) {
    switch (verdict) {
      case 'team_a':
        return 'Team A Superior';
      case 'team_b':
        return 'Team B Superior';
      case 'draw':
        return 'Equal';
      default:
        return verdict;
    }
  }

  // ---------------------------------------------------------------------------
  // Section header helper (MiroFish numbered step card style)
  // ---------------------------------------------------------------------------

  Widget _sectionHeader(String number, String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        children: [
          Container(
            width: 32,
            height: 32,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: Colors.black87,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              number,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 13,
              ),
            ),
          ),
          const SizedBox(width: 10),
          Text(
            title,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: Colors.black87,
            ),
          ),
        ],
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Build
  // ---------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    final debateState = ref.watch(debateProvider);
    final verdicts = debateState.verdicts;

    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        foregroundColor: Colors.black87,
        elevation: 0,
        scrolledUnderElevation: 0.5,
        title: const Text(
          'Verdict',
          style: TextStyle(color: Colors.black87),
        ),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(color: _borderColor, height: 1),
        ),
      ),
      body: debateState.isLoading
          ? const Center(child: CircularProgressIndicator(color: Colors.black87))
          : SingleChildScrollView(
              padding:
                  const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // ---- 01 최종 결과 ----
                  _sectionHeader('01', S.get('final_result')),
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: _borderColor),
                    ),
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      children: [
                        // Verdict badge
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 16, vertical: 8),
                          decoration: BoxDecoration(
                            color: _resultColor(verdicts)
                                .withValues(alpha: 0.1),
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(
                              color: _resultColor(verdicts)
                                  .withValues(alpha: 0.4),
                            ),
                          ),
                          child: Text(
                            _computeResult(verdicts),
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                              color: _resultColor(verdicts),
                            ),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Debate ${widget.debateId}',
                          style: const TextStyle(
                            fontSize: 13,
                            color: Color(0xFF666666),
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 28),

                  // ---- 02 심판 판결 ----
                  _sectionHeader('02', S.get('judge_verdicts')),
                  if (verdicts.isEmpty)
                    Container(
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: _borderColor),
                      ),
                      padding: const EdgeInsets.all(24),
                      child: const Center(
                        child: Text(
                          'No judge verdicts found.',
                          style: TextStyle(color: Color(0xFF999999)),
                        ),
                      ),
                    )
                  else
                    Wrap(
                      spacing: 12,
                      runSpacing: 12,
                      children: verdicts.map((v) {
                        return SizedBox(
                          width: _cardWidth(context, verdicts.length),
                          child: _JudgeVerdictCard(
                            verdict: v,
                            verdictColor: _verdictColor(v.verdict),
                            verdictLabel: _verdictLabel(v.verdict),
                          ),
                        );
                      }).toList(),
                    ),

                  const SizedBox(height: 28),

                  // ---- 03 다음 단계 ----
                  _sectionHeader('03', S.get('next_steps')),
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: _borderColor),
                    ),
                    padding: const EdgeInsets.all(20),
                    child: Row(
                      children: [
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _continueDebate,
                            icon: const Icon(Icons.replay,
                                color: Colors.black87),
                            label: const Text(
                              'Continue Debate (+5 rounds)',
                              style: TextStyle(color: Colors.black87),
                            ),
                            style: OutlinedButton.styleFrom(
                              padding:
                                  const EdgeInsets.symmetric(vertical: 14),
                              side: const BorderSide(color: Color(0xFFE5E5E5)),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(8),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: FilledButton.icon(
                            onPressed: () =>
                                context.go('/report/${widget.debateId}'),
                            icon: const Icon(Icons.description),
                            label: Text(S.get('view_report')),
                            style: FilledButton.styleFrom(
                              backgroundColor: Colors.black87,
                              foregroundColor: Colors.white,
                              padding:
                                  const EdgeInsets.symmetric(vertical: 14),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(8),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 32),

                  if (debateState.error != null)
                    Text(
                      debateState.error!,
                      style: TextStyle(color: Colors.red.shade600),
                    ),
                ],
              ),
            ),
    );
  }

  /// Calculate card width so that cards display in a row on wide screens.
  double _cardWidth(BuildContext context, int count) {
    final screenWidth = MediaQuery.of(context).size.width;
    final available = screenWidth - 48; // padding
    if (count <= 1) return available;
    if (available > 900) {
      // Display cards side by side (3 per row max).
      final perRow = count > 3 ? 3 : count;
      return (available - (perRow - 1) * 12) / perRow;
    }
    return available; // Stack vertically on narrow screens.
  }
}

// ---------------------------------------------------------------------------
// Judge verdict card (light theme)
// ---------------------------------------------------------------------------

/// Expandable card showing a single judge's verdict with score bars.
class _JudgeVerdictCard extends StatelessWidget {
  final Verdict verdict;
  final Color verdictColor;
  final String verdictLabel;

  static const _borderColor = Color(0xFFE5E5E5);

  const _JudgeVerdictCard({
    required this.verdict,
    required this.verdictColor,
    required this.verdictLabel,
  });

  @override
  Widget build(BuildContext context) {
    // Extract total scores for bar visualization.
    final teamATotal = _extractTotal(verdict.teamAScore);
    final teamBTotal = _extractTotal(verdict.teamBScore);
    final maxScore = teamATotal > teamBTotal ? teamATotal : teamBTotal;

    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _borderColor),
      ),
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Judge name.
          Text(
            verdict.judgeId,
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 16,
              color: Colors.black87,
            ),
          ),
          const SizedBox(height: 8),

          // Verdict label.
          Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: verdictColor.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(4),
              border: Border.all(
                  color: verdictColor.withValues(alpha: 0.3)),
            ),
            child: Text(
              verdictLabel,
              style: TextStyle(
                color: verdictColor,
                fontWeight: FontWeight.w600,
                fontSize: 13,
              ),
            ),
          ),
          const SizedBox(height: 8),

          // Confidence.
          Text(
            'Confidence: ${(verdict.confidence * 100).toStringAsFixed(0)}%',
            style: const TextStyle(
                fontSize: 13, color: Color(0xFF666666)),
          ),
          const SizedBox(height: 12),

          // Score bars.
          _ScoreBar(
            label: 'Team A',
            value: teamATotal,
            maxValue: maxScore,
            color: Colors.blue,
          ),
          const SizedBox(height: 6),
          _ScoreBar(
            label: 'Team B',
            value: teamBTotal,
            maxValue: maxScore,
            color: Colors.red,
          ),

          const SizedBox(height: 12),

          // Expandable details.
          Theme(
            data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
            child: ExpansionTile(
              title: Text(
                S.get('view_details'),
                style: TextStyle(fontSize: 13, color: Colors.black87),
              ),
              tilePadding: EdgeInsets.zero,
              iconColor: Colors.black54,
              collapsedIconColor: Colors.black54,
              children: [
                const SizedBox(height: 4),
                Text(
                  S.get('reasoning'),
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 13,
                    color: Colors.black87,
                  ),
                ),
                const SizedBox(height: 4),
                SelectableText(
                  verdict.reasoning,
                  style: const TextStyle(
                      fontSize: 12, height: 1.5, color: Colors.black87),
                ),
                if (verdict.decisiveEvidences.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  Text(
                    S.get('decisive_evidence'),
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 13,
                      color: Colors.black87,
                    ),
                  ),
                  const SizedBox(height: 4),
                  ...verdict.decisiveEvidences.map((e) {
                    final text = e is Map
                        ? e['content'] ?? e.toString()
                        : e.toString();
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 4),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('- ',
                              style: TextStyle(
                                  fontSize: 12, color: Colors.black87)),
                          Expanded(
                            child: SelectableText(
                              text.toString(),
                              style: const TextStyle(
                                  fontSize: 12,
                                  height: 1.4,
                                  color: Colors.black87),
                            ),
                          ),
                        ],
                      ),
                    );
                  }),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Sum all numeric values in a score map to produce a single total.
  double _extractTotal(Map<String, dynamic> scoreMap) {
    double total = 0;
    for (final val in scoreMap.values) {
      if (val is num) total += val.toDouble();
    }
    return total;
  }
}

// ---------------------------------------------------------------------------
// Horizontal score bar (light theme)
// ---------------------------------------------------------------------------

/// A simple horizontal bar showing a team's score relative to the maximum.
class _ScoreBar extends StatelessWidget {
  final String label;
  final double value;
  final double maxValue;
  final Color color;

  const _ScoreBar({
    required this.label,
    required this.value,
    required this.maxValue,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final fraction =
        maxValue > 0 ? (value / maxValue).clamp(0.0, 1.0) : 0.0;

    return Row(
      children: [
        SizedBox(
          width: 56,
          child: Text(
            label,
            style:
                const TextStyle(fontSize: 12, color: Color(0xFF666666)),
          ),
        ),
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(3),
            child: LinearProgressIndicator(
              value: fraction,
              minHeight: 10,
              backgroundColor: const Color(0xFFF0F0F0),
              valueColor: AlwaysStoppedAnimation(color),
            ),
          ),
        ),
        const SizedBox(width: 8),
        SizedBox(
          width: 36,
          child: Text(
            value.toStringAsFixed(0),
            style:
                const TextStyle(fontSize: 12, color: Color(0xFF666666)),
          ),
        ),
      ],
    );
  }
}

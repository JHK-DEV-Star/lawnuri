import 'package:flutter/material.dart';

import '../models/verdict.dart';

/// Detailed verdict card showing a judge's decision with score comparisons,
/// confidence bar, decisive evidence, and full reasoning.
class VerdictDetailWidget extends StatelessWidget {
  /// The verdict data to display.
  final Verdict verdict;

  /// Display name of the judge who issued this verdict.
  final String judgeName;

  const VerdictDetailWidget({
    super.key,
    required this.verdict,
    required this.judgeName,
  });

  /// Determine the verdict badge color.
  Color _verdictColor() {
    final v = verdict.verdict.toLowerCase();
    if (v.contains('team_a') || v.contains('team a')) return Colors.blue;
    if (v.contains('team_b') || v.contains('team b')) return Colors.red;
    if (v.contains('draw') || v.contains('tie')) return Colors.amber;
    return Colors.grey;
  }

  /// Format the verdict string for display.
  String _verdictLabel() {
    final v = verdict.verdict.toLowerCase();
    if (v.contains('team_a') || v.contains('team a')) return 'Team A Wins';
    if (v.contains('team_b') || v.contains('team b')) return 'Team B Wins';
    if (v.contains('draw') || v.contains('tie')) return 'Draw';
    return verdict.verdict;
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      clipBehavior: Clip.antiAlias,
      color: Colors.white,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: const BorderSide(color: Color(0xFFE5E5E5)),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHeader(context),
            const SizedBox(height: 16),

            _buildConfidenceBar(context),
            const SizedBox(height: 20),

            _buildScoreComparison(context),
            const SizedBox(height: 16),

            _buildDecisiveEvidence(context),
            const SizedBox(height: 16),

            _buildReasoning(context),
          ],
        ),
      ),
    );
  }

  /// Build the header with judge name and verdict badge.
  Widget _buildHeader(BuildContext context) {
    final color = _verdictColor();

    return Row(
      children: [
        CircleAvatar(
          radius: 20,
          backgroundColor: Colors.amber.withValues(alpha: 0.2),
          child: const Icon(Icons.gavel, color: Colors.amber, size: 20),
        ),
        const SizedBox(width: 12),

        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                judgeName,
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                  color: Colors.black87,
                ),
              ),
              const Text(
                'Judge',
                style: TextStyle(fontSize: 12, color: Color(0xFF666666)),
              ),
            ],
          ),
        ),

        Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.15),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: color.withValues(alpha: 0.4)),
          ),
          child: Text(
            _verdictLabel(),
            style: TextStyle(
              fontWeight: FontWeight.bold,
              color: color,
              fontSize: 13,
            ),
          ),
        ),
      ],
    );
  }

  /// Build the confidence level indicator bar.
  Widget _buildConfidenceBar(BuildContext context) {
    final percentage = (verdict.confidence * 100).toStringAsFixed(0);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              'Confidence',
              style: TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: Colors.black87),
            ),
            Text(
              '$percentage%',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.bold,
                color: _confidenceColor(verdict.confidence),
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: verdict.confidence,
            minHeight: 8,
            backgroundColor: const Color(0xFFE5E5E5),
            valueColor: AlwaysStoppedAnimation<Color>(
              _confidenceColor(verdict.confidence),
            ),
          ),
        ),
      ],
    );
  }

  /// Choose a color based on confidence level.
  Color _confidenceColor(double confidence) {
    if (confidence >= 0.8) return Colors.green;
    if (confidence >= 0.6) return Colors.amber;
    return Colors.orange;
  }

  /// Build the team score comparison section with horizontal bars.
  Widget _buildScoreComparison(BuildContext context) {
    // Score categories to display.
    const categories = [
      ('Legal Reasoning', 'legal_reasoning'),
      ('Evidence Quality', 'evidence_quality'),
      ('Persuasiveness', 'persuasiveness'),
      ('Rebuttal', 'rebuttal'),
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Score Comparison',
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w600,
            color: Colors.black87,
          ),
        ),
        const SizedBox(height: 4),

        // Legend row.
        Row(
          children: [
            _legendDot(Colors.blue, 'Team A'),
            const SizedBox(width: 16),
            _legendDot(Colors.red, 'Team B'),
          ],
        ),
        const SizedBox(height: 10),

        // Score bars for each category.
        ...categories.map((cat) {
          final label = cat.$1;
          final key = cat.$2;
          final scoreA = _extractScore(verdict.teamAScore, key);
          final scoreB = _extractScore(verdict.teamBScore, key);
          return Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: _buildScoreBar(label, scoreA, scoreB),
          );
        }),
      ],
    );
  }

  /// Extract a numeric score from a team score map by key.
  double _extractScore(Map<String, dynamic> scores, String key) {
    final value = scores[key];
    if (value is num) return value.toDouble();
    return 0.0;
  }

  /// Build a single score comparison bar for one category.
  Widget _buildScoreBar(String label, double scoreA, double scoreB) {
    // Normalize scores to 0-1 range (assuming max 10).
    final maxScore = 10.0;
    final normalizedA = (scoreA / maxScore).clamp(0.0, 1.0);
    final normalizedB = (scoreB / maxScore).clamp(0.0, 1.0);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: const TextStyle(fontSize: 11, color: Color(0xFF666666)),
            ),
            Text(
              '${scoreA.toStringAsFixed(1)} vs ${scoreB.toStringAsFixed(1)}',
              style: const TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w500,
                color: Color(0xFF666666),
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        // Team A bar.
        Row(
          children: [
            const SizedBox(
              width: 48,
              child: Text(
                'A',
                style: TextStyle(
                  fontSize: 10,
                  color: Colors.blue,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(3),
                child: LinearProgressIndicator(
                  value: normalizedA,
                  minHeight: 6,
                  backgroundColor: const Color(0xFFE5E5E5),
                  valueColor:
                      const AlwaysStoppedAnimation<Color>(Colors.blue),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 2),
        // Team B bar.
        Row(
          children: [
            const SizedBox(
              width: 48,
              child: Text(
                'B',
                style: TextStyle(
                  fontSize: 10,
                  color: Colors.red,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(3),
                child: LinearProgressIndicator(
                  value: normalizedB,
                  minHeight: 6,
                  backgroundColor: const Color(0xFFE5E5E5),
                  valueColor:
                      const AlwaysStoppedAnimation<Color>(Colors.red),
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }

  /// Build a small colored dot with label for the legend.
  Widget _legendDot(Color color, String label) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
          ),
        ),
        const SizedBox(width: 4),
        Text(
          label,
          style: const TextStyle(fontSize: 11, color: Color(0xFF666666)),
        ),
      ],
    );
  }

  /// Build the decisive evidence section as an expandable list.
  Widget _buildDecisiveEvidence(BuildContext context) {
    if (verdict.decisiveEvidences.isEmpty) return const SizedBox.shrink();

    return ExpansionTile(
      tilePadding: EdgeInsets.zero,
      title: Row(
        children: [
          const Icon(Icons.star, size: 18, color: Colors.amber),
          const SizedBox(width: 8),
          Text(
            'Decisive Evidence (${verdict.decisiveEvidences.length})',
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: Colors.black87,
            ),
          ),
        ],
      ),
      children: [
        ...verdict.decisiveEvidences.asMap().entries.map((entry) {
          final index = entry.key;
          final evidence = entry.value;
          final text =
              evidence is Map ? (evidence['content'] ?? evidence.toString()) : evidence.toString();

          return Padding(
            padding: const EdgeInsets.only(bottom: 6),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Numbered bullet.
                Container(
                  width: 22,
                  height: 22,
                  alignment: Alignment.center,
                  decoration: BoxDecoration(
                    color: Colors.amber.withValues(alpha: 0.15),
                    shape: BoxShape.circle,
                  ),
                  child: Text(
                    '${index + 1}',
                    style: const TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                      color: Colors.amber,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    text.toString(),
                    style: const TextStyle(fontSize: 12, height: 1.4, color: Colors.black87),
                  ),
                ),
              ],
            ),
          );
        }),
      ],
    );
  }

  /// Build the full reasoning section as an expandable card.
  Widget _buildReasoning(BuildContext context) {
    if (verdict.reasoning.isEmpty) return const SizedBox.shrink();

    return ExpansionTile(
      tilePadding: EdgeInsets.zero,
      title: const Row(
        children: [
          Icon(Icons.description_outlined, size: 18, color: Color(0xFF666666)),
          SizedBox(width: 8),
          Text(
            'Full Reasoning',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: Colors.black87,
            ),
          ),
        ],
      ),
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: const Color(0xFFFAFAFA),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: const Color(0xFFE5E5E5)),
          ),
          child: SelectableText(
            verdict.reasoning,
            style: const TextStyle(
              fontSize: 13,
              height: 1.6,
              color: Color(0xFF666666),
            ),
          ),
        ),
      ],
    );
  }
}

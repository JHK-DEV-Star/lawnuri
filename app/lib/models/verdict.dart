/// Model representing a judge's verdict on a debate.
class Verdict {
  final String judgeId;
  final String judgeName;
  final String verdict;
  final double confidence;
  final Map<String, dynamic> teamAScore;
  final Map<String, dynamic> teamBScore;
  final List<dynamic> decisiveEvidences;
  final String reasoning;

  Verdict({
    required this.judgeId,
    required this.judgeName,
    required this.verdict,
    required this.confidence,
    required this.teamAScore,
    required this.teamBScore,
    required this.decisiveEvidences,
    required this.reasoning,
  });

  /// Create a Verdict from a JSON map (defensive parsing).
  factory Verdict.fromJson(Map<String, dynamic> json) {
    return Verdict(
      judgeId: json['judge_id'] as String? ?? '',
      judgeName: json['judge_name'] as String? ?? json['judge_id'] as String? ?? '',
      verdict: json['verdict'] as String? ?? 'draw',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.5,
      teamAScore: json['score_team_a'] as Map<String, dynamic>?
          ?? json['team_a_score'] as Map<String, dynamic>?
          ?? {},
      teamBScore: json['score_team_b'] as Map<String, dynamic>?
          ?? json['team_b_score'] as Map<String, dynamic>?
          ?? {},
      decisiveEvidences: json['decisive_evidences'] as List? ?? [],
      reasoning: json['reasoning'] as String? ?? '',
    );
  }

  /// Convert this Verdict to a JSON map.
  Map<String, dynamic> toJson() {
    return {
      'judge_id': judgeId,
      'judge_name': judgeName,
      'verdict': verdict,
      'confidence': confidence,
      'team_a_score': teamAScore,
      'team_b_score': teamBScore,
      'decisive_evidences': decisiveEvidences,
      'reasoning': reasoning,
    };
  }

  /// Human-readable verdict label.
  String get verdictLabel {
    switch (verdict) {
      case 'team_a':
        return 'Team A 승리';
      case 'team_b':
        return 'Team B 승리';
      case 'draw':
        return '무승부';
      default:
        return verdict;
    }
  }
}

/// Model representing a piece of evidence submitted during a debate.
class Evidence {
  final String evidenceId;
  final String content;
  final String sourceType;
  final String sourceDetail;
  final String submittedBy;
  final int round;
  final String speaker;

  Evidence({
    required this.evidenceId,
    required this.content,
    required this.sourceType,
    required this.sourceDetail,
    required this.submittedBy,
    required this.round,
    required this.speaker,
  });

  /// Create an Evidence from a JSON map.
  factory Evidence.fromJson(Map<String, dynamic> json) {
    return Evidence(
      evidenceId: json['evidence_id'] as String,
      content: json['content'] as String,
      sourceType: json['source_type'] as String,
      sourceDetail: json['source_detail'] as String,
      submittedBy: json['submitted_by'] as String,
      round: json['round'] as int,
      speaker: json['speaker'] as String,
    );
  }

  /// Convert this Evidence to a JSON map.
  Map<String, dynamic> toJson() {
    return {
      'evidence_id': evidenceId,
      'content': content,
      'source_type': sourceType,
      'source_detail': sourceDetail,
      'submitted_by': submittedBy,
      'round': round,
      'speaker': speaker,
    };
  }
}

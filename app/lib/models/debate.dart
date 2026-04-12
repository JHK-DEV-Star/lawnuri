import 'agent.dart';

/// Request model for creating a new debate.
class DebateCreate {
  final String situationBrief;
  final String defaultModel;

  DebateCreate({
    required this.situationBrief,
    required this.defaultModel,
  });

  /// Convert to JSON map for the API request.
  Map<String, dynamic> toJson() {
    return {
      'situation_brief': situationBrief,
      'default_model': defaultModel,
    };
  }
}

/// Model representing the analysis of a debate topic.
class DebateAnalysis {
  final String topic;
  final String opinionA;
  final String opinionB;
  final List<String> keyIssues;
  final List<String> teamACautions;
  final List<String> teamBCautions;
  final List<Map<String, dynamic>> parties;
  final List<Map<String, dynamic>> timeline;
  final List<String> causalChain;
  final List<Map<String, dynamic>> keyFacts;
  final Map<String, String> focusPoints;
  final List<String> missingInformation;

  DebateAnalysis({
    required this.topic,
    required this.opinionA,
    required this.opinionB,
    required this.keyIssues,
    this.teamACautions = const [],
    this.teamBCautions = const [],
    this.parties = const [],
    this.timeline = const [],
    this.causalChain = const [],
    this.keyFacts = const [],
    this.focusPoints = const {},
    this.missingInformation = const [],
  });

  /// Create a DebateAnalysis from a JSON map.
  factory DebateAnalysis.fromJson(Map<String, dynamic> json) {
    return DebateAnalysis(
      topic: json['topic'] as String? ?? '',
      opinionA: json['opinion_a'] as String? ?? '',
      opinionB: json['opinion_b'] as String? ?? '',
      keyIssues: List<String>.from(json['key_issues'] as List? ?? []),
      teamACautions: List<String>.from(json['team_a_cautions'] as List? ?? []),
      teamBCautions: List<String>.from(json['team_b_cautions'] as List? ?? []),
      parties: (json['parties'] as List?)
          ?.map((e) => Map<String, dynamic>.from(e as Map))
          .toList() ?? [],
      timeline: (json['timeline'] as List?)
          ?.map((e) => Map<String, dynamic>.from(e as Map))
          .toList() ?? [],
      causalChain: List<String>.from(json['causal_chain'] as List? ?? []),
      keyFacts: (json['key_facts'] as List?)
          ?.map((e) => Map<String, dynamic>.from(e as Map))
          .toList() ?? [],
      focusPoints: (json['focus_points'] as Map?)
          ?.map((k, v) => MapEntry(k.toString(), v.toString())) ?? {},
      missingInformation: List<String>.from(json['missing_information'] as List? ?? []),
    );
  }

  /// Convert this DebateAnalysis to a JSON map.
  Map<String, dynamic> toJson() {
    return {
      'topic': topic,
      'opinion_a': opinionA,
      'opinion_b': opinionB,
      'key_issues': keyIssues,
      'team_a_cautions': teamACautions,
      'team_b_cautions': teamBCautions,
      'parties': parties,
      'timeline': timeline,
      'causal_chain': causalChain,
      'key_facts': keyFacts,
      'focus_points': focusPoints,
      'missing_information': missingInformation,
    };
  }
}

/// Model representing the full configuration of a debate session.
class DebateConfig {
  final String debateId;
  final String situationBrief;
  final DebateAnalysis analysis;
  final String defaultModel;
  final List<AgentProfile> agents;
  final String status;
  final int minRounds;
  final int maxRounds;
  final int currentRound;

  DebateConfig({
    required this.debateId,
    required this.situationBrief,
    required this.analysis,
    required this.defaultModel,
    required this.agents,
    required this.status,
    required this.minRounds,
    required this.maxRounds,
    required this.currentRound,
  });

  /// Create a DebateConfig from a JSON map.
  factory DebateConfig.fromJson(Map<String, dynamic> json) {
    return DebateConfig(
      debateId: json['debate_id'] as String,
      situationBrief: json['situation_brief'] as String,
      analysis: DebateAnalysis.fromJson(json['analysis'] as Map<String, dynamic>),
      defaultModel: json['default_model'] as String,
      agents: (json['agents'] as List)
          .map((a) => AgentProfile.fromJson(a as Map<String, dynamic>))
          .toList(),
      status: json['status'] as String,
      minRounds: json['min_rounds'] as int,
      maxRounds: json['max_rounds'] as int,
      currentRound: json['current_round'] as int,
    );
  }

  /// Convert this DebateConfig to a JSON map.
  Map<String, dynamic> toJson() {
    return {
      'debate_id': debateId,
      'situation_brief': situationBrief,
      'analysis': analysis.toJson(),
      'default_model': defaultModel,
      'agents': agents.map((a) => a.toJson()).toList(),
      'status': status,
      'min_rounds': minRounds,
      'max_rounds': maxRounds,
      'current_round': currentRound,
    };
  }

  /// Create a copy with optional field overrides.
  DebateConfig copyWith({
    String? debateId,
    String? situationBrief,
    DebateAnalysis? analysis,
    String? defaultModel,
    List<AgentProfile>? agents,
    String? status,
    int? minRounds,
    int? maxRounds,
    int? currentRound,
  }) {
    return DebateConfig(
      debateId: debateId ?? this.debateId,
      situationBrief: situationBrief ?? this.situationBrief,
      analysis: analysis ?? this.analysis,
      defaultModel: defaultModel ?? this.defaultModel,
      agents: agents ?? this.agents,
      status: status ?? this.status,
      minRounds: minRounds ?? this.minRounds,
      maxRounds: maxRounds ?? this.maxRounds,
      currentRound: currentRound ?? this.currentRound,
    );
  }
}

/// Model representing a single entry in the debate log.
class DebateLogEntry {
  final int round;
  final String team;
  final String speaker;
  final String statement;
  final List<dynamic> evidence;
  final List<Map<String, dynamic>> internalDiscussion;
  final DateTime timestamp;
  final String entryType;
  final int? elapsedSeconds;
  final Map<String, dynamic>? tokenUsage;

  DebateLogEntry({
    required this.round,
    required this.team,
    required this.speaker,
    required this.statement,
    required this.evidence,
    this.internalDiscussion = const [],
    required this.timestamp,
    this.entryType = 'statement',
    this.elapsedSeconds,
    this.tokenUsage,
  });

  /// Create a DebateLogEntry from a JSON map.
  factory DebateLogEntry.fromJson(Map<String, dynamic> json) {
    return DebateLogEntry(
      round: json['round'] as int,
      team: json['team'] as String,
      speaker: json['speaker'] as String,
      statement: json['statement'] as String,
      evidence: List<dynamic>.from(json['evidence'] as List? ?? []),
      internalDiscussion: List<Map<String, dynamic>>.from(
        (json['internal_discussion'] as List?)?.map((e) => e is Map ? Map<String, dynamic>.from(e) : <String, dynamic>{}) ?? [],
      ),
      timestamp: json['timestamp'] != null
          ? (DateTime.tryParse(json['timestamp'] as String) ?? DateTime.now())
          : DateTime.now(),
      entryType: json['entry_type'] as String? ?? 'statement',
      elapsedSeconds: json['elapsed_seconds'] as int?,
      tokenUsage: json['token_usage'] is Map
          ? Map<String, dynamic>.from(json['token_usage'] as Map)
          : null,
    );
  }

  /// Convert this DebateLogEntry to a JSON map.
  Map<String, dynamic> toJson() {
    return {
      'round': round,
      'team': team,
      'speaker': speaker,
      'statement': statement,
      'evidence': evidence,
      'internal_discussion': internalDiscussion,
      'timestamp': timestamp.toIso8601String(),
    };
  }
}

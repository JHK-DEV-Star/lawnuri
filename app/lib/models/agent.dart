/// Model representing a debate agent profile.
class AgentProfile {
  final String agentId;
  final String name;
  final String role;
  final String? team;
  final String specialty;
  final String personality;
  final String debateStyle;
  final String background;
  final String? llmOverride;

  AgentProfile({
    required this.agentId,
    required this.name,
    required this.role,
    this.team,
    required this.specialty,
    required this.personality,
    required this.debateStyle,
    required this.background,
    this.llmOverride,
  });

  /// Create an AgentProfile from a JSON map.
  factory AgentProfile.fromJson(Map<String, dynamic> json) {
    return AgentProfile(
      agentId: json['agent_id'] as String,
      name: json['name'] as String,
      role: json['role'] as String,
      team: json['team'] as String?,
      specialty: json['specialty'] as String,
      personality: json['personality'] as String,
      debateStyle: json['debate_style'] as String,
      background: json['background'] as String,
      llmOverride: json['llm_override'] as String?,
    );
  }

  /// Convert this AgentProfile to a JSON map.
  Map<String, dynamic> toJson() {
    return {
      'agent_id': agentId,
      'name': name,
      'role': role,
      'team': team,
      'specialty': specialty,
      'personality': personality,
      'debate_style': debateStyle,
      'background': background,
      'llm_override': llmOverride,
    };
  }

  /// Create a copy with optional field overrides.
  AgentProfile copyWith({
    String? agentId,
    String? name,
    String? role,
    String? team,
    String? specialty,
    String? personality,
    String? debateStyle,
    String? background,
    String? llmOverride,
  }) {
    return AgentProfile(
      agentId: agentId ?? this.agentId,
      name: name ?? this.name,
      role: role ?? this.role,
      team: team ?? this.team,
      specialty: specialty ?? this.specialty,
      personality: personality ?? this.personality,
      debateStyle: debateStyle ?? this.debateStyle,
      background: background ?? this.background,
      llmOverride: llmOverride ?? this.llmOverride,
    );
  }
}

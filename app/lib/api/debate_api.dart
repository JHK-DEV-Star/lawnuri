import 'package:dio/dio.dart';
import 'api_client.dart';
import '../models/debate.dart';
import '../models/agent.dart';
import '../models/verdict.dart';

/// API client for debate-related endpoints.
class DebateApi {
  final Dio _dio = ApiClient().dio;

  /// List all saved debate sessions.
  Future<List<Map<String, dynamic>>> listDebates() async {
    final response = await _dio.get('/api/debate/list');
    return List<Map<String, dynamic>>.from(response.data as List);
  }

  /// Delete a debate and all associated data.
  Future<void> deleteDebate(String debateId) async {
    await _dio.delete('/api/debate/$debateId');
  }

  /// Create a new debate session.
  Future<Map<String, dynamic>> createDebate(
    String situationBrief,
    String defaultModel,
  ) async {
    final response = await _dio.post(
      '/api/debate/create',
      data: {
        'situation_brief': situationBrief,
        'default_model': defaultModel,
      },
    );
    return response.data as Map<String, dynamic>;
  }

  /// Analyze the debate topic and generate opposing viewpoints.
  Future<DebateAnalysis> analyzeDebate(String debateId) async {
    final response = await _dio.post('/api/debate/$debateId/analyze');
    final data = response.data as Map<String, dynamic>;
    return DebateAnalysis.fromJson(
        data.containsKey('analysis') ? data['analysis'] as Map<String, dynamic> : data);
  }

  /// Generate AI agent profiles for the debate.
  Future<List<AgentProfile>> generateAgents(String debateId) async {
    final response = await _dio.post('/api/debate/$debateId/agents/generate');
    final data = response.data;
    final List list;
    if (data is Map) {
      list = (data['agents'] as List?) ?? [];
    } else {
      list = data as List;
    }
    return list
        .map((a) => AgentProfile.fromJson(a as Map<String, dynamic>))
        .toList();
  }

  /// Get the current list of agents for a debate.
  Future<List<AgentProfile>> getAgents(String debateId) async {
    final response = await _dio.get('/api/debate/$debateId/agents');
    final data = response.data;
    final List list;
    if (data is Map) {
      list = (data['agents'] as List?) ?? [];
    } else {
      list = data as List;
    }
    return list
        .map((a) => AgentProfile.fromJson(a as Map<String, dynamic>))
        .toList();
  }

  /// Update an agent's properties.
  Future<AgentProfile> updateAgent(
    String debateId,
    String agentId,
    Map<String, dynamic> updates,
  ) async {
    final response = await _dio.put(
      '/api/debate/$debateId/agents/$agentId',
      data: updates,
    );
    final data = response.data as Map<String, dynamic>;
    return AgentProfile.fromJson(
        data.containsKey('agent') ? data['agent'] as Map<String, dynamic> : data);
  }

  /// Start the debate simulation.
  Future<Map<String, dynamic>> startDebate(String debateId) async {
    final response = await _dio.post('/api/debate/$debateId/start');
    return response.data as Map<String, dynamic>;
  }

  /// Get the current status of a debate.
  Future<Map<String, dynamic>> getStatus(String debateId) async {
    final response = await _dio.get('/api/debate/$debateId/status');
    return response.data as Map<String, dynamic>;
  }

  /// Get the debate log entries starting from a specific index.
  Future<List<DebateLogEntry>> getLog(
    String debateId, {
    int fromIndex = 0,
  }) async {
    final response = await _dio.get(
      '/api/debate/$debateId/log',
      queryParameters: {'from_index': fromIndex},
    );
    final data = response.data;
    final List list;
    if (data is Map) {
      list = (data['entries'] as List?) ?? [];
    } else {
      list = data as List;
    }
    return list
        .map((e) => DebateLogEntry.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// Get the debate argument graph data.
  Future<Map<String, dynamic>> getDebateGraph(String debateId) async {
    final response = await _dio.get('/api/debate/$debateId/graph');
    return response.data as Map<String, dynamic>;
  }

  /// Send an interrupt/hint to a team during the debate.
  Future<void> interrupt(
    String debateId, {
    String targetTeam = 'team_a',
    String content = '',
    String type = 'hint',
  }) async {
    await _dio.post(
      '/api/debate/$debateId/interrupt',
      data: {
        'target_team': targetTeam,
        'content': content,
        'type': type,
      },
    );
  }

  /// Pause a running debate.
  Future<void> pauseDebate(String debateId) async {
    await _dio.post('/api/debate/$debateId/pause');
  }

  /// Update debate configuration (model overrides).
  Future<void> updateConfig(
    String debateId, {
    String? defaultModel,
    Map<String, String>? agentOverrides,
  }) async {
    final data = <String, dynamic>{};
    if (defaultModel != null) data['default_model'] = defaultModel;
    if (agentOverrides != null) data['agent_overrides'] = agentOverrides;
    await _dio.put('/api/debate/$debateId/config', data: data);
  }

  /// Resume a paused debate.
  Future<Map<String, dynamic>> resumeDebate(String debateId) async {
    final response = await _dio.post('/api/debate/$debateId/resume');
    return response.data as Map<String, dynamic>;
  }

  /// Stop a debate permanently.
  Future<void> stopDebate(String debateId) async {
    await _dio.post('/api/debate/$debateId/stop');
  }

  /// Get the verdicts for a completed debate.
  Future<List<Verdict>> getVerdicts(String debateId) async {
    final response = await _dio.get('/api/debate/$debateId/verdict');
    final data = response.data;
    final List list;
    if (data is Map) {
      list = (data['verdicts'] as List?) ?? [];
    } else {
      list = data as List;
    }
    return list
        .map((v) => Verdict.fromJson(v as Map<String, dynamic>))
        .toList();
  }

  /// Extend a debate by adding more rounds.
  Future<Map<String, dynamic>> extendDebate(
    String debateId, {
    int additionalRounds = 5,
  }) async {
    final response = await _dio.post(
      '/api/debate/$debateId/extend',
      data: {'additional_rounds': additionalRounds},
    );
    return response.data as Map<String, dynamic>;
  }
}

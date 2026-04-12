import 'dart:async';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../api/debate_api.dart';
import '../models/debate.dart';
import '../models/agent.dart';
import '../models/verdict.dart';
import '../services/debate_ws.dart';

/// State model for the active debate session.
class DebateState {
  final String? debateId;
  final String status;
  final List<DebateLogEntry> log;
  final List<AgentProfile> agents;
  final DebateAnalysis? analysis;
  final int currentRound;
  final int maxRounds;
  final List<Verdict> verdicts;
  final String? error;
  final bool isLoading;
  final String? currentPhase;
  final String? currentTeam;
  final int discussionProgress;
  final int discussionTotal;
  final String teamAName;
  final String teamBName;

  const DebateState({
    this.debateId,
    this.status = 'idle',
    this.log = const [],
    this.agents = const [],
    this.analysis,
    this.currentRound = 0,
    this.maxRounds = 0,
    this.verdicts = const [],
    this.error,
    this.isLoading = false,
    this.currentPhase,
    this.currentTeam,
    this.discussionProgress = 0,
    this.discussionTotal = 30,
    this.teamAName = 'Team A',
    this.teamBName = 'Team B',
  });

  /// Create a copy with optional field overrides.
  ///
  /// [clearPhase] / [clearTeam] — set to true to explicitly clear
  /// currentPhase / currentTeam back to null (since passing null for a
  /// nullable parameter is ambiguous with "not provided").
  DebateState copyWith({
    String? debateId,
    String? status,
    List<DebateLogEntry>? log,
    List<AgentProfile>? agents,
    DebateAnalysis? analysis,
    int? currentRound,
    int? maxRounds,
    List<Verdict>? verdicts,
    String? error,
    bool? isLoading,
    String? currentPhase,
    String? currentTeam,
    int? discussionProgress,
    int? discussionTotal,
    String? teamAName,
    String? teamBName,
    bool clearPhase = false,
    bool clearTeam = false,
  }) {
    return DebateState(
      debateId: debateId ?? this.debateId,
      status: status ?? this.status,
      log: log ?? this.log,
      agents: agents ?? this.agents,
      analysis: analysis ?? this.analysis,
      currentRound: currentRound ?? this.currentRound,
      maxRounds: maxRounds ?? this.maxRounds,
      verdicts: verdicts ?? this.verdicts,
      error: error,
      isLoading: isLoading ?? this.isLoading,
      currentPhase: clearPhase ? null : (currentPhase ?? this.currentPhase),
      currentTeam: clearTeam ? null : (currentTeam ?? this.currentTeam),
      discussionProgress: discussionProgress ?? this.discussionProgress,
      discussionTotal: discussionTotal ?? this.discussionTotal,
      teamAName: teamAName ?? this.teamAName,
      teamBName: teamBName ?? this.teamBName,
    );
  }
}

/// StateNotifier that manages the debate lifecycle and live polling.
class DebateNotifier extends StateNotifier<DebateState> {
  final DebateApi _api = DebateApi();
  Timer? _pollingTimer;
  DebateWebSocket? _ws;
  StreamSubscription<Map<String, dynamic>>? _wsSub;
  bool _isPolling = false;
  bool _pendingPoll = false; // poll requested while another was in-flight

  /// Polling interval — slower when WebSocket is active (fallback only).
  static const Duration _pollInterval = Duration(seconds: 5);
  static const Duration _pollIntervalWithWs = Duration(seconds: 15);

  DebateNotifier() : super(const DebateState());

  /// Create a new debate session.
  Future<void> createDebate(String situationBrief, String defaultModel) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final result = await _api.createDebate(situationBrief, defaultModel);
      state = state.copyWith(
        debateId: result['debate_id'] as String,
        status: 'created',
        isLoading: false,
      );
    } catch (e) {
      state = state.copyWith(error: e.toString(), isLoading: false);
    }
  }

  /// Analyze the debate topic.
  Future<void> analyzeDebate() async {
    if (state.debateId == null) return;
    state = state.copyWith(isLoading: true, error: null);
    try {
      final analysis = await _api.analyzeDebate(state.debateId!);
      state = state.copyWith(analysis: analysis, isLoading: false);
    } catch (e) {
      state = state.copyWith(error: e.toString(), isLoading: false);
    }
  }

  /// Generate agents for the debate.
  Future<void> generateAgents() async {
    if (state.debateId == null) return;
    state = state.copyWith(isLoading: true, error: null);
    try {
      final agents = await _api.generateAgents(state.debateId!);
      state = state.copyWith(agents: agents, isLoading: false);
    } catch (e) {
      state = state.copyWith(error: e.toString(), isLoading: false);
    }
  }

  /// Load agents from the backend.
  Future<void> loadAgents() async {
    if (state.debateId == null) return;
    try {
      final agents = await _api.getAgents(state.debateId!);
      state = state.copyWith(agents: agents);
    } catch (e) {
      state = state.copyWith(error: e.toString());
    }
  }

  /// Update a specific agent's properties.
  Future<void> updateAgent(String agentId, Map<String, dynamic> updates) async {
    if (state.debateId == null) return;
    try {
      final updated = await _api.updateAgent(state.debateId!, agentId, updates);
      final updatedAgents = state.agents.map((a) {
        return a.agentId == agentId ? updated : a;
      }).toList();
      state = state.copyWith(agents: updatedAgents);
    } catch (e) {
      state = state.copyWith(error: e.toString());
    }
  }

  /// Start the debate and begin polling for live updates.
  Future<void> startDebate() async {
    if (state.debateId == null) return;
    state = state.copyWith(isLoading: true, error: null);
    try {
      await _api.startDebate(state.debateId!);
      state = state.copyWith(status: 'running', isLoading: false);
      _startPolling();
    } catch (e) {
      state = state.copyWith(error: e.toString(), isLoading: false);
    }
  }

  /// Connect WebSocket for real-time updates, with HTTP polling as fallback.
  void _startPolling() {
    _stopPolling();
    _connectWebSocket();
    final interval = _ws != null ? _pollIntervalWithWs : _pollInterval;
    _pollingTimer = Timer.periodic(interval, (_) => _poll());
  }

  /// Stop polling and disconnect WebSocket.
  void _stopPolling() {
    _pollingTimer?.cancel();
    _pollingTimer = null;
    _pendingPoll = false;
    _disconnectWebSocket();
  }

  /// Connect WebSocket to the debate endpoint.
  void _connectWebSocket() {
    if (state.debateId == null) return;
    _disconnectWebSocket();
    _ws = DebateWebSocket(debateId: state.debateId!);
    _ws!.connect();
    _wsSub = _ws!.events.listen(
      _handleWsEvent,
      onDone: _onWsFailed,
      onError: (_) => _onWsFailed(),
    );
  }

  /// Called when WebSocket permanently fails — switch to faster polling.
  void _onWsFailed() {
    _disconnectWebSocket();
    // Only restart polling if debate is still active
    final s = state.status;
    if (s == 'running' || s == 'paused' || s == 'extended') {
      _pollingTimer?.cancel();
      _pollingTimer = Timer.periodic(_pollInterval, (_) => _poll());
    }
  }

  /// Disconnect WebSocket.
  void _disconnectWebSocket() {
    _wsSub?.cancel();
    _wsSub = null;
    _ws?.dispose();
    _ws = null;
  }

  /// Handle an incoming WebSocket event.
  void _handleWsEvent(Map<String, dynamic> event) {
    final type = event['type'] as String? ?? '';
    switch (type) {
      case 'node_complete':
        if (_isPolling) {
          _pendingPoll = true;
        } else {
          _poll();
        }
        break;
      case 'discussion_message':
        // Update discussion progress from WebSocket data
        final turn = event['turn'] as int?;
        final total = event['total'] as int?;
        if (turn != null && total != null) {
          state = state.copyWith(
            discussionProgress: turn + 1,
            discussionTotal: total,
            currentPhase: 'discussing',
            currentTeam: event['team'] as String? ?? state.currentTeam,
          );
        }
        break;
      case 'phase_change':
        state = state.copyWith(
          currentPhase: event['phase'] as String? ?? state.currentPhase,
        );
        break;
    }
  }

  /// Fetch the latest debate status and log entries.
  Future<void> _poll() async {
    if (state.debateId == null) return;
    if (_isPolling) return; // prevent concurrent polls
    _isPolling = true;
    try {
      final statusData = await _api.getStatus(state.debateId!);
      final newStatus = statusData['status'] as String? ?? state.status;
      final currentRound = statusData['current_round'] as int? ?? state.currentRound;
      final maxRounds = statusData['max_rounds'] as int? ?? state.maxRounds;
      final currentPhase = statusData['current_phase'] as String?;
      final currentTeam = statusData['current_team'] as String?;
      final discussionProgress = statusData['discussion_progress'] as int? ?? state.discussionProgress;
      final discussionTotal = statusData['discussion_total'] as int? ?? state.discussionTotal;
      final teamAName = statusData['team_a_name'] as String? ?? state.teamAName;
      final teamBName = statusData['team_b_name'] as String? ?? state.teamBName;

      // Parse analysis if not yet loaded
      DebateAnalysis? pollAnalysis;
      if (state.analysis == null) {
        final ad = statusData['analysis'];
        if (ad is Map<String, dynamic>) {
          pollAnalysis = DebateAnalysis.fromJson(ad);
        }
      }

      // Fetch new log entries starting from the current count.
      final newEntries = await _api.getLog(
        state.debateId!,
        fromIndex: state.log.length,
      );

      // Skip update if nothing changed.
      if (newStatus == state.status && newEntries.isEmpty &&
          currentPhase == state.currentPhase && currentTeam == state.currentTeam) {
        return;
      }

      final updatedLog = [...state.log, ...newEntries];

      state = state.copyWith(
        status: newStatus,
        currentRound: currentRound,
        maxRounds: maxRounds,
        log: updatedLog,
        currentPhase: currentPhase,
        currentTeam: currentTeam,
        analysis: pollAnalysis ?? state.analysis,
        discussionProgress: discussionProgress,
        discussionTotal: discussionTotal,
        teamAName: teamAName,
        teamBName: teamBName,
      );

      if (newStatus == 'completed' || newStatus == 'finished'
          || newStatus == 'stopped' || newStatus == 'error') {
        // Clear transient phase/team indicators
        state = state.copyWith(clearPhase: true, clearTeam: true);
        _stopPolling();
        if (newStatus == 'completed' || newStatus == 'finished') {
          await loadVerdicts();
        }
      }
    } catch (e) {
      // ignore
    } finally {
      _isPolling = false;
      if (_pendingPoll) {
        _pendingPoll = false;
        Future.microtask(() => _poll());
      }
    }
  }

  /// Send an interrupt/hint to a team.
  Future<void> interrupt({
    String targetTeam = 'team_a',
    String content = '',
    String type = 'hint',
  }) async {
    if (state.debateId == null) return;
    try {
      await _api.interrupt(
        state.debateId!,
        targetTeam: targetTeam,
        content: content,
        type: type,
      );
    } catch (e) {
      state = state.copyWith(error: e.toString());
    }
  }

  /// Pause the debate.
  Future<void> pauseDebate() async {
    if (state.debateId == null) return;
    try {
      await _api.pauseDebate(state.debateId!);
      state = state.copyWith(status: 'paused');
      // Keep polling alive — backend may still be finishing in-flight work
      // (e.g. completing a current LLM call). Polling will auto-stop when
      // the next poll sees a terminal status.
    } catch (e) {
      state = state.copyWith(error: e.toString());
    }
  }

  /// Resume a paused debate.
  Future<void> resumeDebate() async {
    if (state.debateId == null) return;
    try {
      await _api.resumeDebate(state.debateId!);
      state = state.copyWith(status: 'running');
      _startPolling();
    } catch (e) {
      state = state.copyWith(error: e.toString());
    }
  }

  /// Stop the debate permanently.
  Future<void> stopDebate() async {
    if (state.debateId == null) return;
    try {
      await _api.stopDebate(state.debateId!);
      state = state.copyWith(status: 'stopped');
      _stopPolling();
    } catch (e) {
      state = state.copyWith(error: e.toString());
    }
  }

  /// Extend the debate by adding more rounds.
  Future<void> extendDebate({int additionalRounds = 5}) async {
    if (state.debateId == null) return;
    try {
      await _api.extendDebate(state.debateId!, additionalRounds: additionalRounds);
    } catch (e) {
      state = state.copyWith(error: e.toString());
    }
  }

  /// Load verdicts from the backend.
  Future<void> loadVerdicts() async {
    if (state.debateId == null) return;
    try {
      final verdicts = await _api.getVerdicts(state.debateId!);
      state = state.copyWith(verdicts: verdicts);
    } catch (e) {
      state = state.copyWith(error: e.toString());
    }
  }

  /// Load an existing debate by ID (e.g., for navigation).
  Future<void> loadDebate(String debateId) async {
    state = state.copyWith(debateId: debateId, isLoading: true, error: null);
    try {
      final statusData = await _api.getStatus(debateId);
      final status = statusData['status'] as String? ?? 'unknown';
      final currentRound = statusData['current_round'] as int? ?? 0;
      final maxRounds = statusData['max_rounds'] as int? ?? 0;

      // Parse analysis from status response
      DebateAnalysis? analysis;
      final analysisData = statusData['analysis'];
      if (analysisData is Map<String, dynamic>) {
        analysis = DebateAnalysis.fromJson(analysisData);
      }

      final log = await _api.getLog(debateId);
      final agents = await _api.getAgents(debateId);

      state = state.copyWith(
        status: status,
        currentRound: currentRound,
        maxRounds: maxRounds,
        log: log,
        agents: agents,
        analysis: analysis,
        currentPhase: statusData['current_phase'] as String?,
        currentTeam: statusData['current_team'] as String?,
        teamAName: statusData['team_a_name'] as String? ?? 'Team A',
        teamBName: statusData['team_b_name'] as String? ?? 'Team B',
        isLoading: false,
      );

      // Start polling if debate is currently running.
      if (status == 'running') {
        _startPolling();
      }

      // Load verdicts if debate completed naturally (not manual stop).
      if (['completed', 'finished'].contains(status)) {
        await loadVerdicts();
      }
    } catch (e) {
      state = state.copyWith(error: e.toString(), isLoading: false);
    }
  }

  /// Clear the error state.
  void clearError() {
    state = state.copyWith(error: null);
  }

  /// Reset the entire debate state.
  void reset() {
    _stopPolling(); // also disconnects WebSocket
    state = const DebateState();
  }

  @override
  void dispose() {
    _stopPolling(); // also disconnects WebSocket
    super.dispose();
  }
}

/// Global provider for the debate state.
final debateProvider = StateNotifierProvider<DebateNotifier, DebateState>(
  (ref) => DebateNotifier(),
);

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:file_picker/file_picker.dart';
import 'package:desktop_drop/desktop_drop.dart';

import '../providers/debate_provider.dart';
import '../providers/settings_provider.dart';
import '../api/debate_api.dart';
import '../api/rag_api.dart';
import '../l10n/app_strings.dart';

/// Metadata for a user-selected file awaiting upload.
class _UploadedFile {
  final PlatformFile file;
  String pool;
  _UploadedFile({required this.file}) : pool = 'common';
}

class HomeScreen extends ConsumerStatefulWidget {
  const HomeScreen({super.key});

  @override
  ConsumerState<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends ConsumerState<HomeScreen> {
  final TextEditingController _briefController = TextEditingController();
  String? _selectedModel;
  final List<_UploadedFile> _uploadedFiles = [];
  bool _isStarting = false;
  bool _showAgentSidebar = false;
  bool _hasText = false;
  bool _dragging = false;
  List<Map<String, dynamic>> _previousDebates = [];
  bool _loadingHistory = false;
  DateTime _lastHistoryLoad = DateTime.fromMillisecondsSinceEpoch(0);

  // Agent configuration
  int _teamACount = 5;
  int _teamBCount = 5;
  int _judgeCount = 3;
  final Map<String, String?> _agentModels = {}; // agentKey → model override

  String get _settingsTeamAName =>
      ref.read(settingsProvider).debateSettings['team_a_name'] as String? ?? 'Team A';
  String get _settingsTeamBName =>
      ref.read(settingsProvider).debateSettings['team_b_name'] as String? ?? 'Team B';

  @override
  void initState() {
    super.initState();
    _briefController.addListener(_onBriefChanged);
    Future.microtask(() {
      ref.read(settingsProvider.notifier).loadAll();
      _loadHistory();
      // Auto-refresh history when debate status changes (e.g., returning from stopped debate)
      ref.listenManual(debateProvider.select((s) => s.status), (prev, next) {
        if (prev != next) {
          Future.microtask(() => _loadHistory());
        }
      });
    });
  }

  void _onBriefChanged() {
    final hasText = _briefController.text.trim().isNotEmpty;
    if (hasText != _hasText) {
      setState(() => _hasText = hasText);
    }
  }

  @override
  void dispose() {
    _briefController.removeListener(_onBriefChanged);
    _briefController.dispose();
    super.dispose();
  }

  Future<void> _loadHistory() async {
    if (_loadingHistory) return; // prevent concurrent loads
    setState(() => _loadingHistory = true);
    try {
      final debates = await DebateApi().listDebates();
      if (mounted) {
        setState(() {
          _previousDebates = debates;
          _lastHistoryLoad = DateTime.now();
        });
      }
    } catch (_) {
      // Ignore — backend may not be ready yet.
    } finally {
      if (mounted) setState(() => _loadingHistory = false);
    }
  }

  Future<void> _confirmDeleteDebate(Map<String, dynamic> debate) async {
    final id = debate['debate_id'] as String?;
    if (id == null) return;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(S.get('delete_sim')),
        content: const Text('이 시뮬레이션을 삭제하시겠습니까?\n삭제된 데이터는 복구할 수 없습니다.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: Text(S.get('cancel'))),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text(S.get('delete'), style: const TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      try {
        await DebateApi().deleteDebate(id);
        if (mounted) {
          setState(() => _previousDebates.removeWhere((d) => d['debate_id'] == id));
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('삭제 실패: $e')),
          );
        }
      }
    }
  }

  void _navigateToDebate(Map<String, dynamic> debate) {
    final id = debate['debate_id'] as String?;
    if (id == null || id.isEmpty) return;
    // Always navigate to debate screen (verdict is shown in step 04)
    context.go('/debate/$id');
  }

  Future<void> _pickFiles() async {
    final result = await FilePicker.platform.pickFiles(
      allowMultiple: true,
      type: FileType.any,
    );
    if (result != null) {
      setState(() {
        for (final file in result.files) {
          _uploadedFiles.add(_UploadedFile(file: file));
        }
      });
    }
  }

  void _removeFile(int index) {
    setState(() => _uploadedFiles.removeAt(index));
  }

  Future<void> _startSearch() async {
    if (_briefController.text.trim().isEmpty || _selectedModel == null) return;

    // Check legal API key is configured
    final settings = ref.read(settingsProvider);
    final legalApi = settings.legalApiSettings;
    final legalKey = legalApi['law_api_key'] as String? ?? '';
    if (legalKey.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(S.get('legal_api_missing')),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    setState(() => _isStarting = true);

    final notifier = ref.read(debateProvider.notifier);
    try {
      await notifier.createDebate(_briefController.text.trim(), _selectedModel!);
      final debateId = ref.read(debateProvider).debateId;
      if (debateId == null) throw Exception('Failed to create debate');

      if (_uploadedFiles.isNotEmpty) {
        final ragApi = RagApi();
        for (final uploaded in _uploadedFiles) {
          if (uploaded.file.path != null) {
            await ragApi.uploadDocument(debateId, uploaded.pool, File(uploaded.file.path!));
          }
        }
      }

      // Navigate immediately — debate screen handles analyze/agents/start
      if (!mounted) return;
      context.go('/debate/$debateId');
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e'), backgroundColor: Colors.red),
      );
    } finally {
      if (mounted) setState(() => _isStarting = false);
    }
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Refresh history when screen becomes active (e.g., returning from debate)
    if (DateTime.now().difference(_lastHistoryLoad).inSeconds > 3) {
      _lastHistoryLoad = DateTime.now();
      Future.microtask(() => _loadHistory());
    }
  }

  @override
  Widget build(BuildContext context) {
    final settings = ref.watch(settingsProvider);
    final debateState = ref.watch(debateProvider);
    final models = settings.availableModels;
    final canStart = _hasText && _selectedModel != null && !_isStarting;

    return Scaffold(
      body: Column(
        children: [
          // --- Header ---
          _buildHeader(context),
          const Divider(height: 1),
          // --- Main content ---
          Expanded(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // LEFT: Workflow info panel
                Expanded(
                  flex: 4,
                  child: _buildLeftPanel(),
                ),
                // Vertical divider
                Container(width: 1, color: const Color(0xFFEAEAEA)),
                // RIGHT: Console panel
                Expanded(
                  flex: 6,
                  child: _buildRightConsole(models, canStart),
                ),
                // Agent sidebar (animated)
                ClipRect(
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 300),
                    curve: Curves.easeInOut,
                    width: _showAgentSidebar ? 321 : 0, // 320 + 1px border
                    child: _showAgentSidebar
                        ? Row(
                            children: [
                              Container(width: 1, color: const Color(0xFFEAEAEA)),
                              Expanded(child: _buildAgentSidebar(debateState, models)),
                            ],
                          )
                        : const SizedBox.shrink(),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Container(
      height: 56,
      padding: const EdgeInsets.symmetric(horizontal: 24),
      color: Colors.white,
      child: Row(
        children: [
          const Spacer(),
          // Agent sidebar toggle
          IconButton(
            icon: Icon(
              Icons.people_outline,
              color: _showAgentSidebar ? const Color(0xFFFF4500) : Colors.black54,
            ),
            tooltip: 'Agents',
            onPressed: () => setState(() => _showAgentSidebar = !_showAgentSidebar),
          ),
          const SizedBox(width: 4),
          IconButton(
            icon: const Icon(Icons.settings_outlined, color: Colors.black54),
            tooltip: 'Settings',
            onPressed: () => context.push('/settings'),
          ),
        ],
      ),
    );
  }

  Widget _buildLeftPanel() {
    const secondary = Color(0xFF666666);

    // If we have previous debates, show history. Otherwise show workflow guide.
    if (_previousDebates.isNotEmpty) {
      return _buildHistoryPanel();
    }

    // Empty state: show workflow guide
    return SingleChildScrollView(
      padding: const EdgeInsets.all(40),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(width: 10, height: 10, decoration: const BoxDecoration(color: Colors.black, shape: BoxShape.rectangle)),
              const SizedBox(width: 8),
              Text(S.get('workflow'), style: const TextStyle(color: secondary, fontSize: 13)),
            ],
          ),
          const SizedBox(height: 24),
          Text(S.get('system_ready'), style: const TextStyle(fontSize: 28, fontWeight: FontWeight.w800, height: 1.2)),
          const SizedBox(height: 8),
          const Text('상황을 입력하고 검색을 시작하세요', style: TextStyle(color: secondary, fontSize: 14)),
          const SizedBox(height: 40),
          Row(
            children: [
              Container(width: 8, height: 8, decoration: BoxDecoration(border: Border.all(color: secondary, width: 1.5), shape: BoxShape.rectangle)),
              const SizedBox(width: 8),
              Text(S.get('analysis_phase'), style: const TextStyle(color: secondary, fontSize: 13)),
            ],
          ),
          const SizedBox(height: 20),
          _buildWorkflowStep('01', S.get('step_analysis'), S.get('step_analysis_desc')),
          _buildWorkflowStep('02', S.get('step_agents'), S.get('step_agents_desc')),
          _buildWorkflowStep('03', S.get('step_debate'), S.get('step_debate_desc')),
          _buildWorkflowStep('04', S.get('step_verdict'), S.get('step_verdict_desc')),
          _buildWorkflowStep('05', S.get('step_report'), S.get('step_report_desc')),
        ],
      ),
    );
  }

  Widget _buildHistoryPanel() {
    const secondary = Color(0xFF666666);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header
        Padding(
          padding: const EdgeInsets.fromLTRB(24, 24, 24, 0),
          child: Row(
            children: [
              Container(width: 10, height: 10, decoration: const BoxDecoration(color: Colors.black, shape: BoxShape.rectangle)),
              const SizedBox(width: 8),
              Text(S.get('prev_simulations'), style: const TextStyle(color: secondary, fontSize: 13)),
              const Spacer(),
              if (_loadingHistory)
                const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
              else
                IconButton(
                  icon: const Icon(Icons.refresh, size: 18, color: secondary),
                  onPressed: _loadHistory,
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(),
                ),
            ],
          ),
        ),
        const SizedBox(height: 16),
        // Debate list
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            itemCount: _previousDebates.length,
            itemBuilder: (context, index) {
              final debate = _previousDebates[index];
              return _buildDebateHistoryCard(debate);
            },
          ),
        ),
      ],
    );
  }

  Widget _buildDebateHistoryCard(Map<String, dynamic> debate) {
    final topic = (debate['topic'] as String?) ?? '';
    final brief = (debate['situation_brief'] as String?) ?? '';
    final status = (debate['status'] as String?) ?? 'unknown';
    final currentRound = debate['current_round'] as int? ?? 0;
    final maxRounds = debate['max_rounds'] as int? ?? 0;
    final createdAt = (debate['created_at'] as String?) ?? '';

    final displayTitle = topic.isNotEmpty ? topic : (brief.isNotEmpty ? brief : 'Untitled');
    final statusColor = _statusColor(status);
    final dateStr = createdAt.length >= 10 ? createdAt.substring(0, 10) : createdAt;

    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: InkWell(
        onTap: () => _navigateToDebate(debate),
        borderRadius: BorderRadius.circular(4),
        child: Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            border: Border.all(color: const Color(0xFFE5E5E5)),
            borderRadius: BorderRadius.circular(4),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Title + delete button
              Row(
                children: [
                  Expanded(
                    child: Text(
                      displayTitle,
                      style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  SizedBox(
                    width: 28, height: 28,
                    child: IconButton(
                      icon: const Icon(Icons.delete_outline, size: 16, color: Color(0xFFAAAAAA)),
                      padding: EdgeInsets.zero,
                      onPressed: () => _confirmDeleteDebate(debate),
                      tooltip: 'Delete',
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              // Round + status + date
              Row(
                children: [
                  Text(
                    'Round $currentRound/$maxRounds',
                    style: const TextStyle(fontSize: 11, color: Color(0xFF999999)),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                    decoration: BoxDecoration(
                      color: statusColor.withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(2),
                    ),
                    child: Text(
                      status,
                      style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: statusColor),
                    ),
                  ),
                  const Spacer(),
                  Text(dateStr, style: const TextStyle(fontSize: 11, color: Color(0xFF999999))),
                ],
              ),
              // Progress bar
              if (maxRounds > 0) ...[
                const SizedBox(height: 6),
                ClipRRect(
                  borderRadius: BorderRadius.circular(2),
                  child: LinearProgressIndicator(
                    value: currentRound / maxRounds,
                    minHeight: 3,
                    backgroundColor: const Color(0xFFF0F0F0),
                    valueColor: AlwaysStoppedAnimation<Color>(statusColor),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Color _statusColor(String status) {
    switch (status) {
      case 'completed': return const Color(0xFF4CAF50);
      case 'running': return const Color(0xFFFF5722);
      case 'paused': return Colors.orange;
      case 'stopped': return Colors.red;
      default: return const Color(0xFF999999);
    }
  }

  Widget _buildWorkflowStep(String num, String title, String desc) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 24),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            num,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w700,
              color: Color(0xFF999999),
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700)),
                const SizedBox(height: 4),
                Text(desc, style: const TextStyle(color: Color(0xFF666666), fontSize: 13)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRightConsole(List<Map<String, dynamic>> models, bool canStart) {
    return Container(
      margin: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        border: Border.all(color: const Color(0xFFCCCCCC)),
      ),
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // --- Section 01: Documents ---
            _buildConsoleHeader('01', S.get('upload_docs'), trailing: S.get('file_types')),
            const SizedBox(height: 12),
            if (_uploadedFiles.isNotEmpty) ...[
              ..._uploadedFiles.asMap().entries.map((e) => _buildFileItem(e.key, e.value)),
              const SizedBox(height: 8),
            ],
            _buildUploadZone(),

            const SizedBox(height: 24),

            // --- Divider: 입력 파라미터 ---
            _buildCenteredDivider(S.get('input_params')),

            const SizedBox(height: 24),

            // --- Section 02: Situation Brief ---
            _buildConsoleHeader('02', S.get('situation_desc'), prefix: '>_'),
            const SizedBox(height: 12),
            Stack(
              children: [
                TextField(
                  controller: _briefController,
                  maxLines: null,
                  minLines: 10,
                  style: const TextStyle(fontSize: 14, height: 1.5),
                  decoration: InputDecoration(
                    hintText: S.get('situation_hint'),
                    alignLabelWithHint: true,
                    contentPadding: EdgeInsets.all(20),
                  ),
                  // No onChanged setState — _onBriefChanged listener handles it
                ),
                // Model badge removed — model selection is in the right panel dropdown.
              ],
            ),

            const SizedBox(height: 16),

            // --- Model selection with pricing ---
            if (models.isEmpty)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: const Color(0xFFFFF3E0),
                  borderRadius: BorderRadius.circular(4),
                  border: Border.all(color: const Color(0xFFFFCC80)),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.warning_amber, color: Color(0xFFFF9800), size: 18),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        S.get('no_models'),
                        style: TextStyle(fontSize: 13, color: Color(0xFF666666)),
                      ),
                    ),
                    TextButton(
                      onPressed: () => context.push('/settings'),
                      child: const Text('Settings'),
                    ),
                  ],
                ),
              )
            else
              ..._buildModelGroups(models),

            const SizedBox(height: 24),

            // --- Start Search button ---
            SizedBox(
              height: 56,
              child: ElevatedButton(
                onPressed: canStart ? _startSearch : null,
                style: ElevatedButton.styleFrom(
                  disabledBackgroundColor: const Color(0xFFE5E5E5),
                  disabledForegroundColor: const Color(0xFF999999),
                ),
                child: _isStarting
                    ? const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)),
                          SizedBox(width: 12),
                          Text('Starting Search...', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
                        ],
                      )
                    : Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(S.get('start_search'), style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700, letterSpacing: 1)),
                          Icon(Icons.arrow_forward, size: 20),
                        ],
                      ),
              ),
            ),

            if (models.isEmpty) ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: const Color(0xFFFFF8E1),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.warning_amber, color: Color(0xFFFF9800), size: 16),
                    const SizedBox(width: 8),
                    Text(
                      S.get('api_key_warning'),
                      style: TextStyle(fontSize: 12, color: Color(0xFF666666)),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildConsoleHeader(String num, String title, {String? trailing, String? prefix}) {
    return Row(
      children: [
        if (prefix != null) ...[
          Text(prefix, style: const TextStyle(fontSize: 13, color: Color(0xFF999999))),
          const SizedBox(width: 6),
        ],
        Text(
          '$num / $title',
          style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600),
        ),
        const Spacer(),
        if (trailing != null)
          Text(trailing, style: const TextStyle(fontSize: 12, color: Color(0xFF999999))),
      ],
    );
  }

  Widget _buildUploadZone() {
    return DropTarget(
      onDragEntered: (_) => setState(() => _dragging = true),
      onDragExited: (_) => setState(() => _dragging = false),
      onDragDone: (details) {
        setState(() => _dragging = false);
        final files = details.files;
        for (final xFile in files) {
          final name = xFile.name;
          final ext = name.contains('.') ? name.split('.').last.toLowerCase() : '';
          if (!{'pdf', 'md', 'txt', 'markdown'}.contains(ext)) continue;
          _uploadedFiles.add(_UploadedFile(
            file: PlatformFile(
              name: name,
              size: 0,
              path: xFile.path,
            ),
          ));
        }
        setState(() {});
      },
      child: InkWell(
        onTap: _pickFiles,
        child: Container(
          height: _uploadedFiles.isEmpty ? 80 : 36,
          decoration: BoxDecoration(
            border: Border.all(
              color: _dragging ? const Color(0xFF4A90D9) : const Color(0xFFCCCCCC),
              width: _dragging ? 2 : 1,
            ),
            borderRadius: BorderRadius.circular(4),
            color: _dragging ? const Color(0xFFE8F0FE) : const Color(0xFFFAFAFA),
          ),
          child: Center(
            child: _uploadedFiles.isEmpty
                ? Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        _dragging ? Icons.file_download : Icons.upload_file,
                        size: 24,
                        color: _dragging ? const Color(0xFF4A90D9) : const Color(0xFF999999),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        S.get('click_upload'),
                        style: TextStyle(
                          color: _dragging ? const Color(0xFF4A90D9) : const Color(0xFF999999),
                          fontSize: 13,
                        ),
                      ),
                    ],
                  )
                : Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.add, size: 16, color: _dragging ? const Color(0xFF4A90D9) : const Color(0xFF999999)),
                      const SizedBox(width: 4),
                      Text(
                        S.get('add_more_files'),
                        style: TextStyle(
                          color: _dragging ? const Color(0xFF4A90D9) : const Color(0xFF999999),
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
          ),
        ),
      ),
    );
  }

  Widget _buildFileItem(int idx, _UploadedFile uploaded) {
    return Container(
      margin: const EdgeInsets.only(bottom: 4),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border.all(color: const Color(0xFFEEEEEE)),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Row(
        children: [
          const Icon(Icons.insert_drive_file, size: 16, color: Color(0xFF666666)),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              uploaded.file.name,
              style: const TextStyle(fontSize: 13),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          DropdownButton<String>(
            value: uploaded.pool,
            underline: const SizedBox.shrink(),
            isDense: true,
            style: const TextStyle(fontSize: 12, color: Color(0xFF666666)),
            items: [
              const DropdownMenuItem(value: 'common', child: Text('Common')),
              DropdownMenuItem(value: 'team_a', child: Text(S.teamName('team_a', teamAName: _settingsTeamAName, teamBName: _settingsTeamBName))),
              DropdownMenuItem(value: 'team_b', child: Text(S.teamName('team_b', teamAName: _settingsTeamAName, teamBName: _settingsTeamBName))),
            ],
            onChanged: (val) {
              if (val != null) setState(() => uploaded.pool = val);
            },
          ),
          IconButton(
            icon: const Icon(Icons.close, size: 16),
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
            onPressed: () => _removeFile(idx),
          ),
        ],
      ),
    );
  }

  Widget _buildCenteredDivider(String text) {
    return Row(
      children: [
        const Expanded(child: Divider(color: Color(0xFFDDDDDD))),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Text(text, style: const TextStyle(color: Color(0xFFBBBBBB), fontSize: 12)),
        ),
        const Expanded(child: Divider(color: Color(0xFFDDDDDD))),
      ],
    );
  }

  List<Widget> _buildModelGroups(List<Map<String, dynamic>> models) {
    // Group models by provider
    final grouped = <String, List<Map<String, dynamic>>>{};
    for (final m in models) {
      final provider = m['provider'] as String? ?? 'unknown';
      grouped.putIfAbsent(provider, () => []).add(m);
    }

    final widgets = <Widget>[];
    for (final entry in grouped.entries) {
      widgets.add(
        Padding(
          padding: const EdgeInsets.only(bottom: 4, top: 8),
          child: Text(
            entry.key.toUpperCase(),
            style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: Color(0xFF999999), letterSpacing: 0.5),
          ),
        ),
      );
      widgets.add(
        Wrap(
          spacing: 8,
          runSpacing: 6,
          children: entry.value.map((m) {
            final id = m['id'] as String? ?? '';
            final name = m['name'] as String? ?? id;
            final outputPrice = (m['output_price'] as num?)?.toDouble() ?? 0;
            final isSelected = _selectedModel == id;
            // Format price: remove trailing zeros (2.0→2, 0.80→0.8)
            String formatPrice(double p) {
              if (p == p.roundToDouble()) return p.toInt().toString();
              final s = p.toString();
              return s.replaceAll(RegExp(r'0+$'), '').replaceAll(RegExp(r'\.$'), '');
            }
            final priceLabel = outputPrice > 0 ? '\$${formatPrice(outputPrice)}/1M' : 'Free';

            return ChoiceChip(
              label: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(name),
                  const SizedBox(width: 6),
                  Text(priceLabel, style: TextStyle(fontSize: 10, color: isSelected ? Colors.white70 : const Color(0xFFAAAAAA))),
                ],
              ),
              selected: isSelected,
              onSelected: (_) => setState(() => _selectedModel = id),
            );
          }).toList(),
        ),
      );
    }
    return widgets;
  }

  Widget _buildAgentSidebar(dynamic debateState, List<Map<String, dynamic>> models) {
    final totalCount = _teamACount + _teamBCount + _judgeCount;

    // Build model name list for dropdowns
    final modelNames = <String>['(Default)', ...models.map((m) => m['id'] as String? ?? '')];

    return Container(
      color: Colors.white,
      child: Column(
        children: [
          // Header
          Container(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                const Text('Agent Config', style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700)),
                const Spacer(),
                Text('$totalCount', style: const TextStyle(fontSize: 13, color: Color(0xFF999999))),
              ],
            ),
          ),
          const Divider(height: 1),
          Expanded(
            child: ListView(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              children: [
                _buildCountableGroup(S.teamName('team_a', teamAName: _settingsTeamAName, teamBName: _settingsTeamBName), _teamACount, 2, 8, Colors.blue,
                    (v) => setState(() => _teamACount = v), 'team_a', modelNames),
                const SizedBox(height: 8),
                _buildCountableGroup(S.teamName('team_b', teamAName: _settingsTeamAName, teamBName: _settingsTeamBName), _teamBCount, 2, 8, Colors.red,
                    (v) => setState(() => _teamBCount = v), 'team_b', modelNames),
                const SizedBox(height: 8),
                _buildCountableGroup('Judges', _judgeCount, 1, 5, Colors.amber,
                    (v) => setState(() => _judgeCount = v), 'judge', modelNames),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCountableGroup(String title, int count, int min, int max,
      Color color, ValueChanged<int> onChanged, String teamKey, List<String> modelNames) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Group header with +/- buttons
        Row(
          children: [
            Container(
              width: 4, height: 16,
              decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(2)),
            ),
            const SizedBox(width: 8),
            Text(title, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
            const Spacer(),
            IconButton(
              icon: const Icon(Icons.remove, size: 16),
              onPressed: count > min ? () => onChanged(count - 1) : null,
              constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
              padding: EdgeInsets.zero,
              iconSize: 16,
              style: IconButton.styleFrom(
                side: BorderSide(color: count > min ? Colors.black26 : Colors.black12),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 8),
              child: Text('$count', style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
            ),
            IconButton(
              icon: const Icon(Icons.add, size: 16),
              onPressed: count < max ? () => onChanged(count + 1) : null,
              constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
              padding: EdgeInsets.zero,
              iconSize: 16,
              style: IconButton.styleFrom(
                side: BorderSide(color: count < max ? Colors.black26 : Colors.black12),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        // Agent cards with model dropdown
        ...List.generate(count, (i) {
          final agentKey = '${teamKey}_$i';
          final label = teamKey == 'judge' ? 'Judge ${i + 1}' : '${title.split(' ').first} ${i + 1}';
          final selectedModel = _agentModels[agentKey];

          return Container(
            margin: const EdgeInsets.only(bottom: 4),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              border: Border.all(color: const Color(0xFFE5E5E5)),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Row(
              children: [
                Container(
                  width: 6, height: 6,
                  decoration: BoxDecoration(color: color.withValues(alpha: 0.6), shape: BoxShape.circle),
                ),
                const SizedBox(width: 8),
                Text(label, style: const TextStyle(fontSize: 12)),
                const Spacer(),
                // Compact model dropdown
                SizedBox(
                  width: 120,
                  height: 24,
                  child: DropdownButton<String?>(
                    value: selectedModel,
                    isExpanded: true,
                    isDense: true,
                    underline: const SizedBox.shrink(),
                    style: const TextStyle(fontSize: 10, color: Color(0xFF666666)),
                    hint: const Text('Default', style: TextStyle(fontSize: 10, color: Color(0xFFAAAAAA))),
                    items: modelNames.map((m) => DropdownMenuItem<String?>(
                      value: m == '(Default)' ? null : m,
                      child: Text(
                        m == '(Default)' ? 'Default' : m.split('/').last,
                        style: const TextStyle(fontSize: 10),
                        overflow: TextOverflow.ellipsis,
                      ),
                    )).toList(),
                    onChanged: (v) => setState(() => _agentModels[agentKey] = v),
                  ),
                ),
              ],
            ),
          );
        }),
      ],
    );
  }

}

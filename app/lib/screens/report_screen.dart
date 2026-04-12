import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:file_picker/file_picker.dart';
import 'package:go_router/go_router.dart';
import 'package:url_launcher/url_launcher.dart';

import '../api/report_api.dart';
import '../l10n/app_strings.dart';
import '../providers/settings_provider.dart';

/// Screen displaying the full debate report with download capability.
/// MiroFish light theme with numbered collapsible sections (01-07).
class ReportScreen extends ConsumerStatefulWidget {
  final String debateId;
  const ReportScreen({super.key, required this.debateId});

  @override
  ConsumerState<ReportScreen> createState() => _ReportScreenState();
}

class _ReportScreenState extends ConsumerState<ReportScreen> {
  final ReportApi _reportApi = ReportApi();

  static const _borderColor = Color(0xFFE5E5E5);

  Map<String, dynamic>? _report;
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadReport();
  }

  /// Fetch the report data from the backend.
  Future<void> _loadReport() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final data = await _reportApi.getReport(widget.debateId);
      setState(() {
        _report = data;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  /// Download the report file to a user-selected location.
  Future<void> _downloadReport() async {
    final savePath = await FilePicker.platform.saveFile(
      dialogTitle: 'Save Report',
      fileName: 'debate_report_${widget.debateId}.pdf',
      type: FileType.any,
    );

    if (savePath == null) return;

    try {
      await _reportApi.downloadReport(widget.debateId, savePath);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Report saved to $savePath')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Download failed: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        foregroundColor: Colors.black87,
        elevation: 0,
        scrolledUnderElevation: 0.5,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.go('/debate/${widget.debateId}'),
        ),
        title: const Text(
          'Report',
          style: TextStyle(color: Colors.black87),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.download, color: Colors.black87),
            tooltip: S.get('download_report'),
            onPressed: _downloadReport,
          ),
        ],
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(color: _borderColor, height: 1),
        ),
      ),
      body: _isLoading
          ? const Center(
              child: CircularProgressIndicator(color: Colors.black87))
          : _error != null
              ? Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.error_outline,
                          size: 48, color: Colors.red.shade600),
                      const SizedBox(height: 12),
                      Text(_error!,
                          style: TextStyle(color: Colors.red.shade600)),
                      const SizedBox(height: 12),
                      OutlinedButton(
                        onPressed: _loadReport,
                        style: OutlinedButton.styleFrom(
                          foregroundColor: Colors.black87,
                          side: const BorderSide(color: Color(0xFFE5E5E5)),
                        ),
                        child: Text(S.get('retry')),
                      ),
                    ],
                  ),
                )
              : _buildReportContent(),
    );
  }

  Widget _buildReportContent() {
    if (_report == null) {
      return const Center(
          child: Text('No report data.',
              style: TextStyle(color: Colors.black87)));
    }

    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // 00 상황 분석
          _buildSituationAnalysis(),
          const SizedBox(height: 20),

          // 01 종합 요약
          _buildExecutiveSummary(),
          const SizedBox(height: 20),

          // 02 핵심 증거
          _buildDecisiveEvidence(),
          const SizedBox(height: 20),

          // 03 심판 판결
          _buildJudgeVerdicts(),
          const SizedBox(height: 20),

          // 04 논증 분석
          _buildArgumentAnalysis(),
          const SizedBox(height: 20),

          // 05 증거 목록
          _buildEvidenceInventory(),
          const SizedBox(height: 20),

          // 06 라운드별 요약
          _buildRoundSummary(),
          const SizedBox(height: 20),

          // 07 권고사항
          _buildRecommendations(),
          const SizedBox(height: 20),

          // 08 토론 기록
          _buildDebateRecord(),

          const SizedBox(height: 32),
        ],
      ),
    );
  }

  Widget _numberedSectionCard({
    required String number,
    required String title,
    required IconData icon,
    required List<Widget> children,
    Widget? trailing,
    bool initiallyExpanded = false,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _borderColor),
      ),
      clipBehavior: Clip.antiAlias,
      child: Theme(
        data: ThemeData(
          dividerColor: Colors.transparent,
          splashColor: Colors.transparent,
          highlightColor: Colors.transparent,
        ),
        child: ExpansionTile(
          initiallyExpanded: initiallyExpanded,
          expandedCrossAxisAlignment: CrossAxisAlignment.start,
          tilePadding:
              const EdgeInsets.symmetric(horizontal: 20, vertical: 4),
          childrenPadding:
              const EdgeInsets.only(left: 20, right: 20, bottom: 20),
          iconColor: Colors.black54,
          collapsedIconColor: Colors.black54,
          title: Row(
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
              Icon(icon, size: 20, color: Colors.black87),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Colors.black87,
                  ),
                ),
              ),
              if (trailing != null) trailing,
            ],
          ),
          children: children,
        ),
      ),
    );
  }

  Widget _buildExecutiveSummary() {
    final summary = _report!['executive_summary'];
    if (summary == null) return const SizedBox.shrink();

    final result = summary is Map
        ? summary['result'] as String? ?? ''
        : summary.toString();
    final text = summary is Map
        ? (summary['summary'] ?? summary['text'] ?? summary['content'] ?? '').toString()
        : summary.toString();

    return _numberedSectionCard(
      number: '01',
      title: S.get('exec_summary'),
      icon: Icons.summarize,
      initiallyExpanded: true,
      trailing: result.isNotEmpty
          ? Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: _resultBadgeColor(result).withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                    color:
                        _resultBadgeColor(result).withValues(alpha: 0.4)),
              ),
              child: Text(
                result,
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                  color: _resultBadgeColor(result),
                ),
              ),
            )
          : null,
      children: [
        SelectableText(
          text,
          style: const TextStyle(color: Colors.black87, fontSize: 14, height: 1.5),
        ),
      ],
    );
  }

  Widget _buildDecisiveEvidence() {
    final evidenceRaw = _report!['decisive_evidence'];
    if (evidenceRaw == null) return const SizedBox.shrink();

    // New categorized Map format
    if (evidenceRaw is Map) {
      const categoryMeta = <String, (String, String, Color)>{
        'precedent': ('Precedents', '판례', Colors.blue),
        'statute': ('Statutes', '법령', Colors.green),
        'statement': ('Statements', '발언', Colors.orange),
      };
      const medalEmojis = ['\u{1F947}', '\u{1F948}', '\u{1F949}']; // 🥇🥈🥉

      final children = <Widget>[];
      for (final catKey in ['precedent', 'statute', 'statement']) {
        // Support both singular (backend) and plural (legacy) keys
        final items = evidenceRaw[catKey] ?? evidenceRaw['${catKey}s'];
        if (items == null || items is! List || items.isEmpty) continue;
        final (engName, korName, accentColor) = categoryMeta[catKey]!;

        // Category header
        children.add(
          Padding(
            padding: const EdgeInsets.only(top: 12, bottom: 8),
            child: Container(
              decoration: BoxDecoration(
                border: Border(left: BorderSide(color: accentColor, width: 3)),
              ),
              padding: const EdgeInsets.only(left: 12),
              child: Text(
                '\u{1F4CB} $engName ($korName)',
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 14,
                  color: Colors.black87,
                ),
              ),
            ),
          ),
        );

        // Evidence items
        for (var i = 0; i < items.length; i++) {
          final item = items[i];
          if (item is! Map) continue;
          final description =
              (item['description'] ?? item['content'] ?? item['text'] ?? '').toString();
          final impact = (item['impact'] ?? '').toString();
          final source = (item['source'] ?? '').toString();
          final team = (item['team'] ?? '').toString();
          final url = (item['url'] as String?) ?? '';
          final medal = i < medalEmojis.length ? medalEmojis[i] : '#${i + 1}';

          children.add(
            Padding(
              padding: const EdgeInsets.only(bottom: 8, left: 8),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(
                    width: 32,
                    child: Text(medal, style: const TextStyle(fontSize: 16)),
                  ),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Description — Impact
                        SelectableText(
                          impact.isNotEmpty
                              ? '$description — Impact: $impact'
                              : description,
                          style: const TextStyle(
                              fontSize: 13, height: 1.4, color: Colors.black87),
                        ),
                        // Source / Team tags
                        if (source.isNotEmpty || team.isNotEmpty)
                          Padding(
                            padding: const EdgeInsets.only(top: 4),
                            child: Wrap(
                              spacing: 8,
                              children: [
                                if (source.isNotEmpty)
                                  _smallTag('Source: $source', Colors.grey.shade200),
                                if (team.isNotEmpty)
                                  _smallTag('Team: $team', Colors.grey.shade200),
                              ],
                            ),
                          ),
                        // URL link
                        if (url.isNotEmpty)
                          Padding(
                            padding: const EdgeInsets.only(top: 4),
                            child: InkWell(
                              onTap: () => launchUrl(Uri.parse(url)),
                              child: const Text(
                                '원문 보기 \u2192',
                                style: TextStyle(
                                  fontSize: 12,
                                  color: Colors.blue,
                                  decoration: TextDecoration.underline,
                                ),
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          );
        }
      }

      if (children.isEmpty) return const SizedBox.shrink();

      return _numberedSectionCard(
        number: '02',
        title: S.get('decisive_evidence'),
        icon: Icons.emoji_events,
        initiallyExpanded: true,
        children: children,
      );
    }

    // Legacy flat List format (backward compatibility)
    if (evidenceRaw is! List || evidenceRaw.isEmpty) {
      return const SizedBox.shrink();
    }
    final evidenceList = evidenceRaw;

    const medals = ['1st', '2nd', '3rd'];

    return _numberedSectionCard(
      number: '02',
      title: S.get('decisive_evidence'),
      icon: Icons.emoji_events,
      initiallyExpanded: true,
      children: [
        ...evidenceList.asMap().entries.map((entry) {
          final idx = entry.key;
          final item = entry.value;
          final content = item is Map
              ? (item['content'] ?? item['description'] ?? item['text'] ?? '').toString()
              : item.toString();
          final medal =
              idx < medals.length ? medals[idx] : '#${idx + 1}';

          return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SizedBox(
                  width: 36,
                  child: Text(
                    medal,
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: idx == 0
                          ? Colors.amber.shade700
                          : idx == 1
                              ? Colors.grey.shade600
                              : idx == 2
                                  ? Colors.brown.shade400
                                  : const Color(0xFF666666),
                    ),
                  ),
                ),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      SelectableText(
                        content.toString(),
                        style: const TextStyle(
                            fontSize: 13,
                            height: 1.4,
                            color: Colors.black87),
                      ),
                      if (item is Map && (item['url'] as String?)?.isNotEmpty == true)
                        Padding(
                          padding: const EdgeInsets.only(top: 4),
                          child: InkWell(
                            onTap: () => launchUrl(Uri.parse(item['url'] as String)),
                            child: const Text(
                              '원문 보기 \u2192',
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.blue,
                                decoration: TextDecoration.underline,
                              ),
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ],
            ),
          );
        }),
      ],
    );
  }

  Widget _smallTag(String label, Color bgColor) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        label,
        style: const TextStyle(fontSize: 11, color: Colors.black54),
      ),
    );
  }

  Widget _buildJudgeVerdicts() {
    final verdicts = _report!['judge_verdicts'];
    if (verdicts == null || verdicts is! List || verdicts.isEmpty) {
      return const SizedBox.shrink();
    }

    return _numberedSectionCard(
      number: '03',
      title: S.get('judge_verdicts'),
      icon: Icons.gavel,
      initiallyExpanded: true,
      children: [
        ...verdicts.map<Widget>((v) {
          if (v is! Map) return const SizedBox.shrink();
          final judgeName = (v['judge_name'] ?? v['judge_id'] ?? 'Judge').toString();
          final verdict = (v['winner'] ?? v['verdict'] ?? '').toString();
          final confidence = v['confidence'];
          final reasoning = (v['reasoning_summary'] ?? v['reasoning'] ?? '').toString();
          final scores = v['scores'] as Map?;

          return Container(
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: _borderColor),
              color: Colors.grey.shade50,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Judge name + verdict badge row
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.amber.shade100,
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        judgeName,
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.bold,
                          color: Colors.amber.shade900,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: _verdictBadgeColor(verdict).withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                          color: _verdictBadgeColor(verdict).withValues(alpha: 0.4),
                        ),
                      ),
                      child: Text(
                        verdict.isNotEmpty ? verdict : 'N/A',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: _verdictBadgeColor(verdict),
                        ),
                      ),
                    ),
                    const Spacer(),
                    if (confidence != null)
                      Text(
                        '${(confidence is num ? (confidence * 100).round() : confidence)}%',
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.black87,
                        ),
                      ),
                  ],
                ),
                // Reasoning
                if (reasoning.isNotEmpty) ...[
                  const SizedBox(height: 10),
                  SelectableText(
                    reasoning,
                    style: const TextStyle(
                      fontSize: 13,
                      height: 1.5,
                      color: Colors.black87,
                    ),
                  ),
                ],
                // Score bars
                if (scores != null) ...[
                  const SizedBox(height: 12),
                  ...['team_a', 'team_b'].map((teamKey) {
                    final ts = scores[teamKey];
                    if (ts == null || ts is! Map) return const SizedBox.shrink();
                    final ds = ref.read(settingsProvider).debateSettings;
                    final label = S.teamName(teamKey, teamAName: ds['team_a_name'] as String?, teamBName: ds['team_b_name'] as String?);
                    final teamColor = teamKey == 'team_a' ? Colors.blue : Colors.red;
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 6),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            label,
                            style: TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                              color: teamColor,
                            ),
                          ),
                          const SizedBox(height: 4),
                          ...['evidence_quality', 'argument_logic', 'persuasiveness', 'rebuttal_effectiveness', 'overall'].map((key) {
                            final score = ts[key];
                            if (score == null) return const SizedBox.shrink();
                            final numScore = score is num ? score.toDouble() : 0.0;
                            return Padding(
                              padding: const EdgeInsets.only(bottom: 3),
                              child: Row(
                                children: [
                                  SizedBox(
                                    width: 100,
                                    child: Text(
                                      key.replaceAll('_', ' '),
                                      style: const TextStyle(fontSize: 11, color: Color(0xFF666666)),
                                    ),
                                  ),
                                  Expanded(
                                    child: ClipRRect(
                                      borderRadius: BorderRadius.circular(3),
                                      child: LinearProgressIndicator(
                                        value: (numScore / 100).clamp(0.0, 1.0),
                                        backgroundColor: Colors.grey.shade200,
                                        valueColor: AlwaysStoppedAnimation<Color>(teamColor.withValues(alpha: 0.7)),
                                        minHeight: 6,
                                      ),
                                    ),
                                  ),
                                  const SizedBox(width: 6),
                                  Text(
                                    '$score',
                                    style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.black87),
                                  ),
                                ],
                              ),
                            );
                          }),
                        ],
                      ),
                    );
                  }),
                ],
              ],
            ),
          );
        }),
      ],
    );
  }

  /// Return a color for a verdict badge based on team.
  Color _verdictBadgeColor(String verdict) {
    final lower = verdict.toLowerCase();
    if (lower.contains('team_a') || lower.contains('team a')) return Colors.blue;
    if (lower.contains('team_b') || lower.contains('team b')) return Colors.red;
    return Colors.grey;
  }

  Widget _buildArgumentAnalysis() {
    final analysis = _report!['argument_analysis'];
    if (analysis == null || analysis is! Map) {
      return const SizedBox.shrink();
    }

    final teamA = analysis['team_a'] as Map?;
    final teamB = analysis['team_b'] as Map?;

    return _numberedSectionCard(
      number: '04',
      title: S.get('argument_analysis'),
      icon: Icons.balance,
      initiallyExpanded: false,
      children: [
        LayoutBuilder(
          builder: (context, constraints) {
            final isWide = constraints.maxWidth > 600;
            final ds = ref.read(settingsProvider).debateSettings;
            final tAName = S.teamName('team_a', teamAName: ds['team_a_name'] as String?, teamBName: ds['team_b_name'] as String?);
            final tBName = S.teamName('team_b', teamAName: ds['team_a_name'] as String?, teamBName: ds['team_b_name'] as String?);
            if (isWide) {
              return Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: _teamAnalysisColumn(
                        tAName, Colors.blue, teamA),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: _teamAnalysisColumn(
                        tBName, Colors.red, teamB),
                  ),
                ],
              );
            }
            return Column(
              children: [
                _teamAnalysisColumn(tAName, Colors.blue, teamA),
                const SizedBox(height: 16),
                _teamAnalysisColumn(tBName, Colors.red, teamB),
              ],
            );
          },
        ),
      ],
    );
  }

  /// Build a column for one team's argument analysis.
  Widget _teamAnalysisColumn(String title, Color color, Map? data) {
    if (data == null) {
      return Text('$title: No data',
          style: TextStyle(color: color));
    }

    final strengths = data['strongest'] as List? ?? data['strengths'] as List? ?? [];
    final weaknesses = data['weakest'] as List? ?? data['weaknesses'] as List? ?? [];
    final missing = data['missing_evidence'] as List? ?? data['missing'] as List? ?? [];

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.2)),
        color: color.withValues(alpha: 0.03),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 15,
              color: color,
            ),
          ),
          const SizedBox(height: 8),
          if (strengths.isNotEmpty) ...[
            Text(
              S.get('strengths'),
              style: TextStyle(
                fontWeight: FontWeight.w600,
                fontSize: 13,
                color: Colors.green,
              ),
            ),
            ..._bulletList(strengths),
            const SizedBox(height: 8),
          ],
          if (weaknesses.isNotEmpty) ...[
            Text(
              S.get('weaknesses'),
              style: TextStyle(
                fontWeight: FontWeight.w600,
                fontSize: 13,
                color: Colors.orange.shade700,
              ),
            ),
            ..._bulletList(weaknesses),
            const SizedBox(height: 8),
          ],
          if (missing.isNotEmpty) ...[
            Text(
              S.get('missing'),
              style: TextStyle(
                fontWeight: FontWeight.w600,
                fontSize: 13,
                color: Colors.red.shade600,
              ),
            ),
            ..._bulletList(missing),
          ],
        ],
      ),
    );
  }

  Widget _buildEvidenceInventory() {
    final inventory = _report!['evidence_inventory'];
    if (inventory == null || inventory is! Map) {
      return const SizedBox.shrink();
    }

    final evidenceItems = _report!['evidence_items'];

    return _numberedSectionCard(
      number: '05',
      title: S.get('evidence_list'),
      icon: Icons.inventory_2,
      initiallyExpanded: false,
      children: [
        Wrap(
          spacing: 16,
          runSpacing: 8,
          children: inventory.entries.expand<Widget>((e) {
            final val = e.value;
            if (val is Map) {
              return val.entries.map<Widget>((nested) {
                return _StatChip(
                    label: nested.key.toString(),
                    value: nested.value.toString());
              });
            }
            return [
              _StatChip(
                  label: e.key.toString(),
                  value: val.toString()),
            ];
          }).toList(),
        ),
        // Actual evidence items chips
        if (evidenceItems != null && evidenceItems is List && evidenceItems.isNotEmpty) ...[
          const SizedBox(height: 16),
          const Divider(height: 1, color: Color(0xFFE5E5E5)),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: evidenceItems.map<Widget>((item) {
              if (item is! Map) return const SizedBox.shrink();
              final sourceType = (item['source_type'] ?? '').toString().toLowerCase();
              final sourceDetail = (item['source_detail'] ?? '').toString();
              final url = (item['url'] ?? '').toString();
              final chipColor = _evidenceTypeColor(sourceType);

              return ActionChip(
                avatar: Icon(
                  _evidenceTypeIcon(sourceType),
                  size: 14,
                  color: chipColor,
                ),
                label: Text(
                  sourceDetail,
                  style: TextStyle(fontSize: 11, color: chipColor),
                ),
                backgroundColor: chipColor.withValues(alpha: 0.08),
                side: BorderSide(color: chipColor.withValues(alpha: 0.3)),
                onPressed: url.isNotEmpty
                    ? () => launchUrl(Uri.parse(url))
                    : null,
              );
            }).toList(),
          ),
        ],
      ],
    );
  }

  /// Return a color based on evidence source type.
  Color _evidenceTypeColor(String sourceType) {
    if (sourceType.contains('statute') || sourceType.contains('legal')) return Colors.green;
    if (sourceType.contains('precedent') || sourceType.contains('court')) return Colors.blue;
    if (sourceType.contains('constitution')) return Colors.purple;
    return Colors.grey;
  }

  /// Return an icon based on evidence source type.
  IconData _evidenceTypeIcon(String sourceType) {
    if (sourceType.contains('statute') || sourceType.contains('legal')) return Icons.gavel;
    if (sourceType.contains('precedent') || sourceType.contains('court')) return Icons.account_balance;
    if (sourceType.contains('constitution')) return Icons.security;
    return Icons.description;
  }

  Widget _buildRoundSummary() {
    final rounds = _report!['debate_flow_summary'] ?? _report!['round_summary'];
    if (rounds == null || rounds is! List || rounds.isEmpty) {
      return const SizedBox.shrink();
    }

    return _numberedSectionCard(
      number: '06',
      title: S.get('round_summary'),
      icon: Icons.timeline,
      initiallyExpanded: false,
      children: [
        ...rounds.asMap().entries.map((entry) {
          final idx = entry.key;
          final round = entry.value;
          final roundNum =
              round is Map ? round['round'] ?? idx + 1 : idx + 1;
          final summary = round is Map
              ? (round['summary'] ?? round['team_a_summary'] ?? round['key_moment'] ?? '').toString()
              : round.toString();

          return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 32,
                  height: 32,
                  alignment: Alignment.center,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Colors.blue.withValues(alpha: 0.1),
                    border: Border.all(
                        color:
                            Colors.blue.withValues(alpha: 0.3)),
                  ),
                  child: Text(
                    '$roundNum',
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 13,
                      color: Colors.blue,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: SelectableText(
                    summary.toString(),
                    style: const TextStyle(
                        fontSize: 13,
                        height: 1.4,
                        color: Colors.black87),
                  ),
                ),
              ],
            ),
          );
        }),
      ],
    );
  }

  Widget _buildRecommendations() {
    final recs = _report!['recommendations'];
    if (recs == null) return const SizedBox.shrink();

    // Handle both list and map formats.
    List<dynamic> items = [];
    if (recs is List) {
      items = recs;
    } else if (recs is Map) {
      // Flatten map values.
      for (final v in recs.values) {
        if (v is List) {
          items.addAll(v);
        } else {
          items.add(v);
        }
      }
    }

    if (items.isEmpty) return const SizedBox.shrink();

    return _numberedSectionCard(
      number: '07',
      title: S.get('recommendations'),
      icon: Icons.lightbulb_outline,
      initiallyExpanded: false,
      children: [
        ..._bulletList(items),
      ],
    );
  }

  /// Build a bulleted list of items.
  List<Widget> _bulletList(List items) {
    return items.map((item) {
      final text = item is Map
          ? item['text'] ?? item.toString()
          : item.toString();
      return Padding(
        padding: const EdgeInsets.only(bottom: 4, left: 8),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '\u2022 ',
              style:
                  TextStyle(fontSize: 13, color: Color(0xFF666666)),
            ),
            Expanded(
              child: SelectableText(
                text.toString(),
                style: const TextStyle(
                    fontSize: 13,
                    height: 1.4,
                    color: Colors.black87),
              ),
            ),
          ],
        ),
      );
    }).toList();
  }

  /// Return a color for the result badge in the executive summary.
  Color _resultBadgeColor(String result) {
    final lower = result.toLowerCase();
    if (lower.contains('team_a') || lower.contains('team a')) {
      return Colors.blue;
    }
    if (lower.contains('team_b') || lower.contains('team b')) {
      return Colors.red;
    }
    return Colors.orange;
  }

  Widget _buildSituationAnalysis() {
    final sit = _report!['situation_analysis'] as Map<String, dynamic>?;
    if (sit == null || sit.isEmpty) return const SizedBox.shrink();

    return _numberedSectionCard(
      number: '00',
      title: S.get('topic_analysis'),
      icon: Icons.search,
      initiallyExpanded: true,
      children: [Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (sit['situation_brief'] != null && (sit['situation_brief'] as String).isNotEmpty) ...[
            Text(S.get('situation_brief_label'),
                style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: Color(0xFF999999))),
            const SizedBox(height: 4),
            SelectableText(sit['situation_brief'] as String,
                style: const TextStyle(fontSize: 13, height: 1.5)),
            const SizedBox(height: 12),
          ],
          if (sit['topic'] != null && (sit['topic'] as String).isNotEmpty) ...[
            Text(sit['topic'] as String,
                style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
          ],
          if (sit['opinion_a'] != null && (sit['opinion_a'] as String).isNotEmpty) ...[
            Row(children: [
              Container(width: 8, height: 8, decoration: const BoxDecoration(color: Colors.blue, shape: BoxShape.circle)),
              const SizedBox(width: 6),
              Expanded(child: SelectableText(sit['opinion_a'] as String,
                  style: const TextStyle(fontSize: 12, color: Colors.blue))),
            ]),
            const SizedBox(height: 4),
          ],
          if (sit['opinion_b'] != null && (sit['opinion_b'] as String).isNotEmpty) ...[
            Row(children: [
              Container(width: 8, height: 8, decoration: const BoxDecoration(color: Colors.red, shape: BoxShape.circle)),
              const SizedBox(width: 6),
              Expanded(child: SelectableText(sit['opinion_b'] as String,
                  style: const TextStyle(fontSize: 12, color: Colors.red))),
            ]),
            const SizedBox(height: 8),
          ],
          for (final issue in sit['key_issues'] as List? ?? [])
            Padding(
              padding: const EdgeInsets.only(left: 8, bottom: 2),
              child: SelectableText('• $issue', style: const TextStyle(fontSize: 12)),
            ),
          if ((sit['parties'] as List?)?.isNotEmpty ?? false) ...[
            const SizedBox(height: 8),
            Text(S.get('parties'), style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
            for (final p in sit['parties'] as List)
              Padding(
                padding: const EdgeInsets.only(left: 8, bottom: 2),
                child: SelectableText(p is Map ? '• ${p['name'] ?? ''} - ${p['role'] ?? ''}' : '• $p',
                    style: const TextStyle(fontSize: 12)),
              ),
          ],
          if ((sit['timeline'] as List?)?.isNotEmpty ?? false) ...[
            const SizedBox(height: 8),
            Text('Timeline', style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
            for (final t in sit['timeline'] as List)
              Padding(
                padding: const EdgeInsets.only(left: 8, bottom: 2),
                child: t is Map
                    ? Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          SelectableText(
                            '• [${t['date'] ?? ''}] ${t['action'] ?? ''}',
                            style: const TextStyle(fontSize: 12),
                          ),
                          if (t['significance'] != null &&
                              t['significance'].toString().isNotEmpty)
                            Padding(
                              padding: const EdgeInsets.only(left: 12),
                              child: SelectableText(
                                t['significance'].toString(),
                                style: const TextStyle(
                                    fontSize: 11, color: Color(0xFF888888)),
                              ),
                            ),
                        ],
                      )
                    : SelectableText('• $t',
                        style: const TextStyle(fontSize: 12)),
              ),
          ],
        ],
      )],
    );
  }

  Widget _buildDebateRecord() {
    final transcript = _report!['transcript'] as List?;
    if (transcript == null || transcript.isEmpty) return const SizedBox.shrink();

    // Group by round
    final Map<int, List<Map<String, dynamic>>> byRound = {};
    for (final entry in transcript) {
      if (entry is! Map) continue;
      final rnd = entry['round'] as int? ?? 0;
      byRound.putIfAbsent(rnd, () => []).add(Map<String, dynamic>.from(entry));
    }
    final rounds = byRound.keys.toList()..sort();

    return _numberedSectionCard(
      number: '08',
      title: S.get('debate_record'),
      icon: Icons.history,
      initiallyExpanded: false,
      children: [Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          for (final rnd in rounds) ...[
            Container(
              padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
              margin: const EdgeInsets.only(bottom: 8),
              decoration: BoxDecoration(
                color: const Color(0xFFF5F5F5),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text('Round $rnd',
                  style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700)),
            ),
            for (final entry in byRound[rnd]!) ...[
              Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(
                          color: (entry['team'] == 'team_a')
                              ? Colors.blue.withAlpha(25)
                              : (entry['team'] == 'team_b')
                                  ? Colors.red.withAlpha(25)
                                  : Colors.amber.withAlpha(25),
                          borderRadius: BorderRadius.circular(3),
                        ),
                        child: Text(
                          '${(entry['team'] as String? ?? '').toUpperCase()} | ${entry['speaker'] ?? ''}',
                          style: TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                            color: (entry['team'] == 'team_a')
                                ? Colors.blue
                                : (entry['team'] == 'team_b')
                                    ? Colors.red
                                    : Colors.amber.shade800,
                          ),
                        ),
                      ),
                    ]),
                    const SizedBox(height: 4),
                    SelectableText(
                      entry['statement'] as String? ?? '',
                      style: const TextStyle(fontSize: 12, height: 1.5),
                    ),
                  ],
                ),
              ),
            ],
            if (rnd != rounds.last) const Divider(height: 16, color: Color(0xFFE5E5E5)),
          ],
        ],
      )],
    );
  }

}

/// Small chip showing a key-value stat pair.
class _StatChip extends StatelessWidget {
  final String label;
  final String value;

  const _StatChip({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding:
          const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFE5E5E5)),
      ),
      child: Column(
        children: [
          Text(
            value,
            style: const TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: Colors.black87,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: const TextStyle(
                fontSize: 11, color: Color(0xFF666666)),
          ),
        ],
      ),
    );
  }
}

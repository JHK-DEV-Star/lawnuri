import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:url_launcher/url_launcher.dart';

import '../api/settings_api.dart';
import '../l10n/app_strings.dart';
import '../providers/settings_provider.dart';

/// Settings screen for managing API keys, debate parameters,
/// and external legal API configuration.
/// All changes auto-save after a short debounce — no Save buttons needed.
class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key});

  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
  static const Color _accent = Color(0xFFFF4500);
  static const Color _cardBorder = Color(0xFFE5E5E5);
  static const Color _inputFill = Color(0xFFFAFAFA);
  static const Color _inputBorder = Color(0xFFDDDDDD);
  static const Color _subtleText = Color(0xFF666666);

  final _openaiKeyCtrl = TextEditingController();
  final _geminiKeyCtrl = TextEditingController();
  final _anthropicKeyCtrl = TextEditingController();
  final _vertexKeyCtrl = TextEditingController();
  final _vertexProjectIdCtrl = TextEditingController();
  final _vertexLocationCtrl = TextEditingController();
  final _customKeyCtrl = TextEditingController();
  final _customBaseUrlCtrl = TextEditingController();
  final _customModelCtrl = TextEditingController();

  final _teamANameCtrl = TextEditingController();
  final _teamBNameCtrl = TextEditingController();
  final _minRoundsCtrl = TextEditingController();
  final _maxRoundsCtrl = TextEditingController();
  final _teamDiscussionTurnsCtrl = TextEditingController();
  final _judgeEarlyStopCtrl = TextEditingController();
  final _chunkSizeCtrl = TextEditingController();
  final _chunkOverlapCtrl = TextEditingController();
  final _ragTopKCtrl = TextEditingController();
  final _teamSizeCtrl = TextEditingController();
  final _judgeCountCtrl = TextEditingController();
  final _maxReviewMoreCtrl = TextEditingController();
  final _acceptRatioCtrl = TextEditingController();
  final _reviewMoreRatioCtrl = TextEditingController();
  double _temperature = 0.7;
  String _selectedLanguage = 'ko';

  final _legalApiKeyCtrl = TextEditingController();
  final _maxApiCallsCtrl = TextEditingController();

  // Provider test results (provider name -> success flag).
  final Map<String, bool?> _testResults = {};

  // Save status for brief "Saved" indicator
  final Map<String, bool> _saveStatus = {};

  // Selected model per provider
  final Map<String, String?> _selectedModels = {};

  // Gemini tier: 'free' (default, no billing) | 'billing' (paid tier)
  String _geminiTier = 'free';

  // Legal API test result
  bool? _legalTestResult;
  bool _legalTestLoading = false;

  // Model lists with pricing: {model: "$output_price/1M"}
  static const Map<String, Map<String, String>> _presetModelPricing = {
    'openai': {
      'gpt-5.4': r'$15/1M',
      'gpt-5.4-mini': r'$4.5/1M',
      'gpt-5.4-nano': r'$1.25/1M',
      'gpt-5.2': r'$14/1M',
      'gpt-5-mini': r'$2/1M',
      'gpt-4.1': r'$8/1M',
      'gpt-4.1-mini': r'$1.6/1M',
      'gpt-4.1-nano': r'$0.4/1M',
      'gpt-4o': r'$10/1M',
      'gpt-4o-mini': r'$0.6/1M',
      'o3': r'$8/1M',
      'o4-mini': r'$4.4/1M',
    },
    // Gemini pricing is tier-dependent — see _geminiTierPricing below.
    // Keep an empty placeholder here so _presetModelPricing.containsKey('gemini')
    // stays true and the model selector still renders.
    'gemini': {},
    'anthropic': {
      'claude-opus-4-6': r'$25/1M',
      'claude-opus-4-5': r'$25/1M',
      'claude-opus-4-1': r'$75/1M',
      'claude-opus-4': r'$75/1M',
      'claude-sonnet-4-6': r'$15/1M',
      'claude-sonnet-4-5': r'$15/1M',
      'claude-sonnet-4': r'$15/1M',
      'claude-haiku-4-5': r'$5/1M',
    },
    'vertex_ai': {
      'gemini-3.1-pro-preview': r'$12/1M',
      'gemini-3.1-flash-lite-preview': r'$1.5/1M',
      'gemini-3-flash-preview': r'$3/1M',
      'gemini-3-pro-preview': r'$12/1M',
      'gemini-2.5-pro': r'$10/1M',
      'gemini-2.5-flash': r'$2.5/1M',
      'gemini-2.5-flash-lite': r'$0.4/1M',
      'gemini-2.0-flash': r'$0.6/1M',
    },
  };

  // Gemini tier-dependent pricing.
  // Free: limited rate, no cost, paid-only models hidden.
  // Billing: full model list with per-1M paid prices.
  static const Map<String, Map<String, String>> _geminiTierPricing = {
    'free': {
      'gemini-3.1-flash-lite-preview': 'Free 15 RPM',
      'gemini-3-flash-preview': 'Free 10 RPM',
      'gemini-2.5-pro': 'Free 5 RPM',
      'gemini-2.5-flash': 'Free 10 RPM',
      'gemini-2.5-flash-lite': 'Free 15 RPM',
      'gemini-2.0-flash': 'Free 15 RPM',
      // gemini-3.1-pro-preview is paid-only, not shown on free tier.
    },
    'billing': {
      'gemini-3.1-pro-preview': r'$12/1M',
      'gemini-3.1-flash-lite-preview': r'$1.5/1M',
      'gemini-3-flash-preview': r'$3/1M',
      'gemini-2.5-pro': r'$10/1M',
      'gemini-2.5-flash': r'$2.5/1M',
      'gemini-2.5-flash-lite': r'$0.4/1M',
      'gemini-2.0-flash': r'$0.4/1M',
    },
  };

  // Provider pricing page URLs
  static const Map<String, String> _pricingUrls = {
    'openai': 'https://openai.com/api/pricing/',
    'gemini': 'https://ai.google.dev/gemini-api/docs/pricing',
    'anthropic': 'https://platform.claude.com/docs/en/about-claude/pricing',
    'vertex_ai': 'https://cloud.google.com/vertex-ai/generative-ai/pricing',
  };

  // Helper to get model list from pricing map. Gemini uses tier-dependent list.
  List<String> _presetModelsFor(String provider) {
    if (provider == 'gemini') {
      return _geminiTierPricing[_geminiTier]?.keys.toList() ?? [];
    }
    return _presetModelPricing[provider]?.keys.toList() ?? [];
  }

  // Helper to get model price string. Gemini uses tier-dependent pricing.
  String _modelPriceFor(String provider, String model) {
    if (provider == 'gemini') {
      return _geminiTierPricing[_geminiTier]?[model] ?? '';
    }
    return _presetModelPricing[provider]?[model] ?? '';
  }

  // API key visibility toggles
  final Map<String, bool> _obscureKeys = {
    'openai': true,
    'gemini': true,
    'anthropic': true,
    'vertex_ai': true,
    'custom': true,
    'law': true,
    'case': true,
  };

  // Debounce timers for auto-save
  final Map<String, Timer> _debounceTimers = {};

  final SettingsApi _api = SettingsApi();

  @override
  void initState() {
    super.initState();
    Future.microtask(() async {
      final notifier = ref.read(settingsProvider.notifier);
      await notifier.loadAll();
      _syncFromState();
      // Load legal API settings separately (stored in legal_api section)
      try {
        final legalSettings = await _api.getLegalApiSettings();
        if (mounted) {
          setState(() {
            _legalApiKeyCtrl.text =
                legalSettings['law_api_key'] as String? ?? '';
            _maxApiCallsCtrl.text =
                '${legalSettings['max_api_calls_per_round'] ?? 10}';
          });
        }
      } catch (_) {
        // legal_api endpoint not available yet, keep defaults
      }
    });
  }

  /// Populate text controllers from the current provider state (initial load only).
  void _syncFromState() {
    final state = ref.read(settingsProvider);
    final providers = state.providers;

    // API keys.
    _openaiKeyCtrl.text =
        (providers['openai'] as Map?)?['api_key'] as String? ?? '';
    _geminiKeyCtrl.text =
        (providers['gemini'] as Map?)?['api_key'] as String? ?? '';
    _anthropicKeyCtrl.text =
        (providers['anthropic'] as Map?)?['api_key'] as String? ?? '';
    _vertexKeyCtrl.text =
        (providers['vertex_ai'] as Map?)?['api_key'] as String? ?? '';
    _vertexProjectIdCtrl.text =
        (providers['vertex_ai'] as Map?)?['project_id'] as String? ?? '';
    _vertexLocationCtrl.text =
        (providers['vertex_ai'] as Map?)?['location'] as String? ?? 'global';
    _customKeyCtrl.text =
        (providers['custom'] as Map?)?['api_key'] as String? ?? '';
    _customBaseUrlCtrl.text =
        (providers['custom'] as Map?)?['base_url'] as String? ?? '';
    _customModelCtrl.text =
        (providers['custom'] as Map?)?['model'] as String? ?? '';

    // Gemini tier: 'free' (default) or 'billing'
    _geminiTier =
        (providers['gemini'] as Map?)?['tier'] as String? ?? 'free';

    // Debate settings.
    final ds = state.debateSettings;
    _teamANameCtrl.text = ds['team_a_name'] as String? ?? 'Team A';
    _teamBNameCtrl.text = ds['team_b_name'] as String? ?? 'Team B';
    _minRoundsCtrl.text = '${ds['min_rounds'] ?? 3}';
    _maxRoundsCtrl.text = '${ds['max_rounds'] ?? 5}';
    _teamDiscussionTurnsCtrl.text = '${ds['team_discussion_turns'] ?? 10}';
    _judgeEarlyStopCtrl.text = '${ds['judge_early_stop_votes'] ?? 2}';
    _chunkSizeCtrl.text = '${ds['chunk_size'] ?? 1500}';
    _chunkOverlapCtrl.text = '${ds['chunk_overlap'] ?? 200}';
    _ragTopKCtrl.text = '${ds['rag_top_k'] ?? 10}';
    _teamSizeCtrl.text = '${ds['team_size'] ?? 5}';
    _judgeCountCtrl.text = '${ds['judge_count'] ?? 3}';
    _maxReviewMoreCtrl.text = '${ds['max_review_more'] ?? 3}';
    _acceptRatioCtrl.text = '${ds['accept_ratio'] ?? 0.4}';
    _reviewMoreRatioCtrl.text = '${ds['review_more_ratio'] ?? 0.6}';
    _temperature = (ds['temperature'] as num?)?.toDouble() ?? 0.7;
    _selectedLanguage = ds['language'] as String? ?? 'ko';

    // Load selected models per provider
    for (final p in ['openai', 'gemini', 'anthropic', 'vertex_ai', 'custom']) {
      final cfg = providers[p] as Map?;
      _selectedModels[p] = cfg?['model'] as String?;
    }

    // Note: Legal API settings are loaded separately in initState
    // from GET /api/settings/legal-api (not from debate settings).

    if (mounted) setState(() {}); // rebuild after loading provider settings
  }

  @override
  void dispose() {
    for (final t in _debounceTimers.values) {
      t.cancel();
    }
    _openaiKeyCtrl.dispose();
    _geminiKeyCtrl.dispose();
    _anthropicKeyCtrl.dispose();
    _vertexKeyCtrl.dispose();
    _vertexProjectIdCtrl.dispose();
    _vertexLocationCtrl.dispose();
    _customKeyCtrl.dispose();
    _customBaseUrlCtrl.dispose();
    _customModelCtrl.dispose();
    _teamANameCtrl.dispose();
    _teamBNameCtrl.dispose();
    _minRoundsCtrl.dispose();
    _maxRoundsCtrl.dispose();
    _teamDiscussionTurnsCtrl.dispose();
    _judgeEarlyStopCtrl.dispose();
    _chunkSizeCtrl.dispose();
    _chunkOverlapCtrl.dispose();
    _ragTopKCtrl.dispose();
    _teamSizeCtrl.dispose();
    _judgeCountCtrl.dispose();
    _legalApiKeyCtrl.dispose();
    _maxApiCallsCtrl.dispose();
    super.dispose();
  }

  void _debounce(String key, VoidCallback action) {
    _debounceTimers[key]?.cancel();
    _debounceTimers[key] = Timer(const Duration(milliseconds: 600), action);
  }

  /// Auto-save a provider's API key (and optional fields) without refreshing UI.
  void _autoSaveProvider(String provider) {
    _debounce('provider_$provider', () async {
      String apiKey;
      String? baseUrl;
      String? model;

      switch (provider) {
        case 'openai':
          apiKey = _openaiKeyCtrl.text.trim();
          break;
        case 'gemini':
          apiKey = _geminiKeyCtrl.text.trim();
          break;
        case 'anthropic':
          apiKey = _anthropicKeyCtrl.text.trim();
          break;
        case 'vertex_ai':
          apiKey = _vertexKeyCtrl.text.trim();
          break;
        case 'custom':
          apiKey = _customKeyCtrl.text.trim();
          baseUrl = _customBaseUrlCtrl.text.trim().isNotEmpty
              ? _customBaseUrlCtrl.text.trim()
              : null;
          model = _customModelCtrl.text.trim().isNotEmpty
              ? _customModelCtrl.text.trim()
              : null;
          break;
        default:
          return;
      }

      try {
        if (provider == 'vertex_ai') {
          await _api.updateProviderRaw(provider, {
            'api_key': _vertexKeyCtrl.text.trim(),
            'enabled': true,
            'project_id': _vertexProjectIdCtrl.text.trim(),
            'location': _vertexLocationCtrl.text.trim().isNotEmpty
                ? _vertexLocationCtrl.text.trim()
                : 'global',
            if (_selectedModels['vertex_ai'] != null)
              'model': _selectedModels['vertex_ai'],
          });
        } else {
          await _api.updateProvider(provider, apiKey,
              baseUrl: baseUrl, model: model ?? _selectedModels[provider]);
        }
        // Reload provider state so status indicator updates
        await ref.read(settingsProvider.notifier).loadProviders();
        // Show brief "Saved" indicator
        if (mounted) {
          setState(() => _saveStatus[provider] = true);
          Future.delayed(const Duration(seconds: 2), () {
            if (mounted) setState(() => _saveStatus.remove(provider));
          });
        }
      } catch (_) {
        // silent — background save
      }
    });
  }

  /// Auto-save debate settings without refreshing UI.
  void _autoSaveDebateSettings() {
    _debounce('debate', () async {
      final settings = <String, dynamic>{
        'team_a_name': _teamANameCtrl.text,
        'team_b_name': _teamBNameCtrl.text,
        'min_rounds': int.tryParse(_minRoundsCtrl.text) ?? 3,
        'max_rounds': int.tryParse(_maxRoundsCtrl.text) ?? 5,
        'team_discussion_turns':
            int.tryParse(_teamDiscussionTurnsCtrl.text) ?? 30,
        'judge_early_stop_votes':
            int.tryParse(_judgeEarlyStopCtrl.text) ?? 2,
        'chunk_size': int.tryParse(_chunkSizeCtrl.text) ?? 1500,
        'chunk_overlap': int.tryParse(_chunkOverlapCtrl.text) ?? 200,
        'rag_top_k': int.tryParse(_ragTopKCtrl.text) ?? 10,
        'team_size': int.tryParse(_teamSizeCtrl.text) ?? 5,
        'judge_count': int.tryParse(_judgeCountCtrl.text) ?? 3,
        'max_review_more': int.tryParse(_maxReviewMoreCtrl.text) ?? 3,
        'accept_ratio': double.tryParse(_acceptRatioCtrl.text) ?? 0.4,
        'review_more_ratio': double.tryParse(_reviewMoreRatioCtrl.text) ?? 0.6,
        'temperature': _temperature,
        'language': _selectedLanguage,
      };
      try {
        await _api.updateDebateSettings(settings);
      } catch (_) {}
    });
  }

  /// Auto-save legal API settings via dedicated endpoint.
  void _autoSaveExternalApi() {
    _debounce('external_api', () async {
      final key = _legalApiKeyCtrl.text.trim();
      final settings = <String, dynamic>{
        'law_api_key': key,
        'precedent_api_key': key, // same OC key for both
        'max_api_calls_per_round':
            int.tryParse(_maxApiCallsCtrl.text) ?? 10,
      };
      try {
        await _api.updateLegalApiSettings(settings);
      } catch (_) {}
    });
  }

  Future<void> _testProvider(String provider) async {
    setState(() => _testResults[provider] = null);
    try {
      final result =
          await ref.read(settingsProvider.notifier).testProvider(provider);
      final ok = (result['status'] ?? '') == 'ok';
      if (mounted) {
        setState(() => _testResults[provider] = ok);
      }
      // Auto-detect Gemini tier from backend response.
      if (ok && provider == 'gemini') {
        final detected = result['tier'] as String?;
        if (detected == 'free' || detected == 'billing') {
          if (mounted) setState(() => _geminiTier = detected!);
          // Reload providers + models so the pricing labels reflect new tier.
          await ref.read(settingsProvider.notifier).loadProviders();
          await ref.read(settingsProvider.notifier).loadModels(force: true);
        }
      }
    } catch (_) {
      if (mounted) setState(() => _testResults[provider] = false);
    }
  }

  InputDecoration _inputDecoration(String label, {IconData? icon}) {
    return InputDecoration(
      labelText: label,
      labelStyle: const TextStyle(color: _subtleText),
      prefixIcon: icon != null ? Icon(icon, color: _subtleText) : null,
      filled: true,
      fillColor: _inputFill,
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: const BorderSide(color: _inputBorder),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: const BorderSide(color: _accent, width: 1.5),
      ),
      contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
    );
  }

  /// Gemini Free/Billing tier toggle — shown next to the help icon in the card header.
  Widget _buildGeminiTierToggle() {
    return SegmentedButton<String>(
      segments: [
        ButtonSegment(
          value: 'free',
          label: Text(S.get('tier_free'), style: const TextStyle(fontSize: 11)),
        ),
        ButtonSegment(
          value: 'billing',
          label: Text(S.get('tier_billing'), style: const TextStyle(fontSize: 11)),
        ),
      ],
      selected: {_geminiTier},
      showSelectedIcon: false,
      onSelectionChanged: (sel) async {
        setState(() => _geminiTier = sel.first);
        await _saveGeminiTier();
        // 토글 변경에 따라 모델 목록/가격이 달라지므로 전체 재로드
        await ref.read(settingsProvider.notifier).loadProviders();
        await ref.read(settingsProvider.notifier).loadModels(force: true);
      },
      style: ButtonStyle(
        visualDensity: VisualDensity.compact,
        tapTargetSize: MaterialTapTargetSize.shrinkWrap,
        padding: WidgetStateProperty.all(
          const EdgeInsets.symmetric(horizontal: 10, vertical: 0),
        ),
        textStyle: WidgetStateProperty.all(
          const TextStyle(fontSize: 11),
        ),
      ),
    );
  }

  /// Save current Gemini tier selection to the backend.
  Future<void> _saveGeminiTier() async {
    try {
      await _api.updateProviderRaw('gemini', {
        'api_key': _geminiKeyCtrl.text.trim(),
        'enabled': true,
        'tier': _geminiTier,
        if (_selectedModels['gemini'] != null) 'model': _selectedModels['gemini'],
      });
    } catch (_) {
      // silent — background save
    }
  }

  Widget _providerStatus(String provider) {
    final providers = ref.watch(settingsProvider).providers;
    final config = providers[provider] as Map?;
    final hasKey = (config?['api_key'] as String?)?.isNotEmpty == true;

    final saved = _saveStatus[provider] == true;
    final testResult = _testResults[provider];

    // Determine status: test result > key presence > not configured
    final IconData icon;
    final String label;
    final Color color;

    if (testResult == true) {
      icon = Icons.check_circle;
      label = S.get('connected');
      color = Colors.green;
    } else if (testResult == false) {
      icon = Icons.error;
      label = S.get('failed');
      color = Colors.red;
    } else if (hasKey) {
      icon = Icons.info_outline;
      label = S.get('key_set');
      color = Colors.orange;
    } else {
      icon = Icons.cancel;
      label = S.get('not_configured');
      color = Colors.grey;
    }

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (saved) ...[
          const Icon(Icons.check, size: 14, color: Colors.green),
          const SizedBox(width: 2),
          Text(S.get('saved'), style: const TextStyle(fontSize: 11, color: Colors.green)),
          const SizedBox(width: 8),
        ],
        Icon(icon, size: 16, color: color),
        const SizedBox(width: 4),
        Text(label, style: TextStyle(fontSize: 12, color: color)),
      ],
    );
  }

  Widget _buildProviderCard({
    required String provider,
    required String label,
    required TextEditingController keyCtrl,
    TextEditingController? baseUrlCtrl,
    TextEditingController? modelCtrl,
    String? hint,
    Widget? trailingLabel,
  }) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      color: Colors.white,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: const BorderSide(color: _cardBorder),
      ),
      elevation: 0,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      label,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: Colors.black87,
                      ),
                    ),
                    if (_pricingUrls.containsKey(provider)) ...[
                      const SizedBox(width: 4),
                      IconButton(
                        icon: const Icon(Icons.help_outline, size: 16, color: _subtleText),
                        onPressed: () async {
                          final uri = Uri.parse(_pricingUrls[provider]!);
                          if (await canLaunchUrl(uri)) {
                            await launchUrl(uri, mode: LaunchMode.externalApplication);
                          }
                        },
                        tooltip: S.get('pricing_info'),
                        constraints: const BoxConstraints(),
                        padding: const EdgeInsets.all(4),
                      ),
                    ],
                    if (trailingLabel != null) ...[
                      const SizedBox(width: 6),
                      trailingLabel,
                    ],
                  ],
                ),
                _providerStatus(provider),
              ],
            ),
            if (hint != null) ...[
              const SizedBox(height: 4),
              Text(hint, style: const TextStyle(color: _subtleText, fontSize: 11)),
            ],
            const SizedBox(height: 12),
            TextField(
              controller: keyCtrl,
              obscureText: _obscureKeys[provider] ?? true,
              enableInteractiveSelection: true,
              enableSuggestions: false,
              autocorrect: false,
              onChanged: (_) {
                setState(() {});
                _autoSaveProvider(provider);
              },
              style: const TextStyle(color: Colors.black87),
              decoration:
                  _inputDecoration(S.get('api_key'), icon: Icons.key).copyWith(
                suffixIcon: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (keyCtrl.text.isNotEmpty)
                      IconButton(
                        icon: const Icon(Icons.clear,
                            size: 18, color: Color(0xFF999999)),
                        onPressed: () {
                          setState(() => keyCtrl.clear());
                          _autoSaveProvider(provider);
                        },
                        tooltip: S.get('clear'),
                      ),
                    IconButton(
                      icon: Icon(
                        (_obscureKeys[provider] ?? true)
                            ? Icons.visibility_off
                            : Icons.visibility,
                        color: const Color(0xFF999999),
                        size: 20,
                      ),
                      onPressed: () => setState(() => _obscureKeys[provider] =
                          !(_obscureKeys[provider] ?? true)),
                      tooltip: (_obscureKeys[provider] ?? true)
                          ? S.get('show_key')
                          : S.get('hide_key'),
                    ),
                  ],
                ),
              ),
            ),
            if (baseUrlCtrl != null) ...[
              const SizedBox(height: 8),
              TextField(
                controller: baseUrlCtrl,
                onChanged: (_) {
                _autoSaveProvider(provider);
              },
                style: const TextStyle(color: Colors.black87),
                decoration: _inputDecoration(S.get('base_url'), icon: Icons.link)
                    .copyWith(
                      helperText: S.get('custom_base_url_helper'),
                      helperMaxLines: 2,
                    ),
              ),
            ],
            if (modelCtrl != null) ...[
              const SizedBox(height: 8),
              TextField(
                controller: modelCtrl,
                onChanged: (_) {
                _autoSaveProvider(provider);
              },
                style: const TextStyle(color: Colors.black87),
                decoration:
                    _inputDecoration(S.get('model_name'), icon: Icons.smart_toy)
                        .copyWith(
                          helperText: S.get('custom_model_helper'),
                          helperMaxLines: 2,
                        ),
              ),
            ],
            // Model selection chips with pricing (preset providers only)
            if (_presetModelPricing.containsKey(provider)) ...[
              const SizedBox(height: 12),
              Row(
                children: [
                  Text(S.get('model_label'),
                      style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600,
                          color: _subtleText)),
                  const SizedBox(width: 6),
                  Text(S.get('per_1m_tokens'),
                      style: const TextStyle(fontSize: 10, color: Color(0xFF999999))),
                ],
              ),
              const SizedBox(height: 6),
              Wrap(
                spacing: 6,
                runSpacing: 4,
                children: _presetModelsFor(provider).map((model) {
                  final selected = _selectedModels[provider] == model;
                  final price = _modelPriceFor(provider, model);
                  return ChoiceChip(
                    label: Text(
                      '$model  $price',
                      style: const TextStyle(fontSize: 11),
                    ),
                    selected: selected,
                    selectedColor: Colors.black,
                    backgroundColor: Colors.white,
                    labelStyle: TextStyle(
                      color: selected ? Colors.white : Colors.black87,
                      fontSize: 11,
                    ),
                    side: BorderSide(
                      color: selected ? Colors.black : _cardBorder,
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(4),
                    ),
                    onSelected: (sel) {
                      setState(() => _selectedModels[provider] = sel ? model : null);
                      _autoSaveProvider(provider);
                    },
                  );
                }).toList(),
              ),
            ],
            const SizedBox(height: 12),
            Row(
              children: [
                OutlinedButton.icon(
                  onPressed: () => _testProvider(provider),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.black87,
                    side: const BorderSide(color: _cardBorder),
                  ),
                  icon: _testResults[provider] == null &&
                          _testResults.containsKey(provider)
                      ? const SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: _accent,
                          ),
                        )
                      : const Icon(Icons.wifi_tethering, size: 16),
                  label: Text(S.get('test')),
                ),
                const SizedBox(width: 8),
                if (_testResults[provider] == true)
                  Text(
                    S.get('success'),
                    style: const TextStyle(color: Colors.green, fontSize: 12),
                  )
                else if (_testResults[provider] == false)
                  Text(
                    S.get('failed'),
                    style: const TextStyle(color: Colors.red, fontSize: 12),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _numericField(String label, TextEditingController ctrl,
      {VoidCallback? onChanged, String? hint}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: TextField(
        controller: ctrl,
        keyboardType: TextInputType.number,
        onChanged: (_) => onChanged?.call(),
        style: const TextStyle(color: Colors.black87),
        decoration: _inputDecoration(label).copyWith(helperText: hint),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final settings = ref.watch(settingsProvider);

    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: Text(
          S.get('settings'),
          style: const TextStyle(color: Colors.black87, fontWeight: FontWeight.w600),
        ),
        backgroundColor: Colors.white,
        elevation: 0,
        iconTheme: const IconThemeData(color: Colors.black87),
      ),
      body: GestureDetector(
        onTap: () {
          // Restore keyboard focus after Win+V or other native popups
          FocusScope.of(context).unfocus();
        },
        behavior: HitTestBehavior.translucent,
        child: settings.isLoading
          ? const Center(
              child: CircularProgressIndicator(color: _accent),
            )
          : SingleChildScrollView(
              padding:
                  const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // ===== Language Selection (TOP) =====
                  Container(
                    margin: const EdgeInsets.only(bottom: 24),
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      border: Border.all(color: const Color(0xFFE5E5E5)),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Row(
                      children: [
                        Text(S.get('language'), style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
                        const SizedBox(width: 16),
                        DropdownButton<String>(
                          value: _selectedLanguage,
                          underline: const SizedBox.shrink(),
                          items: const [
                            DropdownMenuItem(value: 'ko', child: Text('한국어')),
                            DropdownMenuItem(value: 'en', child: Text('English')),
                            DropdownMenuItem(value: 'ja', child: Text('日本語')),
                            DropdownMenuItem(value: 'zh', child: Text('中文')),
                            DropdownMenuItem(value: 'es', child: Text('Español')),
                            DropdownMenuItem(value: 'fr', child: Text('Français')),
                            DropdownMenuItem(value: 'de', child: Text('Deutsch')),
                            DropdownMenuItem(value: 'pt', child: Text('Português')),
                            DropdownMenuItem(value: 'vi', child: Text('Tiếng Việt')),
                            DropdownMenuItem(value: 'th', child: Text('ภาษาไทย')),
                          ],
                          onChanged: (v) {
                            if (v != null) {
                              S.setLanguage(v);  // update in-memory locale immediately
                              setState(() => _selectedLanguage = v);
                              _autoSaveDebateSettings();
                            }
                          },
                        ),
                      ],
                    ),
                  ),
                  // ---- External Legal API (shown first in the API Keys area) ----
                  _sectionHeader(S.get('external_legal_api')),
                  Padding(
                    padding: const EdgeInsets.only(left: 4, bottom: 8),
                    child: InkWell(
                      onTap: () async {
                        final uri = Uri.parse('https://open.law.go.kr');
                        if (await canLaunchUrl(uri)) {
                          await launchUrl(uri,
                              mode: LaunchMode.externalApplication);
                        }
                      },
                      child: Text(
                        S.get('legal_api_link_text'),
                        style: const TextStyle(
                          color: Color(0xFF0066CC),
                          fontSize: 12,
                          decoration: TextDecoration.underline,
                        ),
                      ),
                    ),
                  ),
                  Card(
                    color: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                      side: const BorderSide(color: _cardBorder),
                    ),
                    elevation: 0,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        children: [
                          Padding(
                            padding: const EdgeInsets.only(bottom: 8),
                            child: Text(
                              S.get('legal_api_same_key_note'),
                              style: const TextStyle(color: _subtleText, fontSize: 12),
                            ),
                          ),
                          TextField(
                            controller: _legalApiKeyCtrl,
                            obscureText: _obscureKeys['law'] ?? true,
                            enableInteractiveSelection: true,
                            enableSuggestions: false,
                            autocorrect: false,
                            onChanged: (_) => _autoSaveExternalApi(),
                            style: const TextStyle(color: Colors.black87),
                            decoration: _inputDecoration(
                                    S.get('legal_api_key_label'), icon: Icons.key)
                                .copyWith(
                              suffixIcon: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  if (_legalApiKeyCtrl.text.isNotEmpty)
                                    IconButton(
                                      icon: const Icon(Icons.clear,
                                          size: 18, color: Color(0xFF999999)),
                                      onPressed: () {
                                        setState(() => _legalApiKeyCtrl.clear());
                                        _autoSaveExternalApi();
                                      },
                                      tooltip: S.get('clear'),
                                    ),
                                  IconButton(
                                    icon: Icon(
                                        (_obscureKeys['law'] ?? true)
                                            ? Icons.visibility_off
                                            : Icons.visibility,
                                        color: const Color(0xFF999999),
                                        size: 20),
                                    onPressed: () => setState(() =>
                                        _obscureKeys['law'] =
                                            !(_obscureKeys['law'] ?? true)),
                                  ),
                                ],
                              ),
                            ),
                          ),
                          const SizedBox(height: 12),
                          Row(
                            children: [
                              OutlinedButton.icon(
                                onPressed: _legalTestLoading ? null : () async {
                                  setState(() {
                                    _legalTestLoading = true;
                                    _legalTestResult = null;
                                  });
                                  try {
                                    final result = await _api.testLegalApi();
                                    setState(() {
                                      _legalTestResult = (result['status'] ?? '') == 'ok';
                                      _legalTestLoading = false;
                                    });
                                  } catch (_) {
                                    setState(() {
                                      _legalTestResult = false;
                                      _legalTestLoading = false;
                                    });
                                  }
                                },
                                style: OutlinedButton.styleFrom(
                                  foregroundColor: Colors.black87,
                                  side: const BorderSide(color: _cardBorder),
                                ),
                                icon: _legalTestLoading
                                    ? const SizedBox(
                                        width: 14, height: 14,
                                        child: CircularProgressIndicator(
                                            strokeWidth: 2, color: _accent))
                                    : const Icon(Icons.wifi_tethering, size: 16),
                                label: Text(S.get('test')),
                              ),
                              const SizedBox(width: 8),
                              if (_legalTestResult == true)
                                Text(S.get('success'),
                                    style: const TextStyle(color: Colors.green, fontSize: 12))
                              else if (_legalTestResult == false)
                                Text(S.get('failed'),
                                    style: const TextStyle(color: Colors.red, fontSize: 12)),
                            ],
                          ),
                          const SizedBox(height: 8),
                          _numericField(S.get('max_api_calls'),
                              _maxApiCallsCtrl,
                              onChanged: _autoSaveExternalApi),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 24),

                  // ---- API Keys ----
                  _sectionHeader(S.get('api_keys')),
                  Padding(
                    padding: const EdgeInsets.only(left: 2, bottom: 8),
                    child: Text(
                      S.get('auto_save_note'),
                      style: const TextStyle(color: _subtleText, fontSize: 12),
                    ),
                  ),
                  _buildProviderCard(
                    provider: 'openai',
                    label: 'OpenAI',
                    keyCtrl: _openaiKeyCtrl,
                  ),
                  _buildProviderCard(
                    provider: 'gemini',
                    label: 'Gemini (Google AI Studio)',
                    keyCtrl: _geminiKeyCtrl,
                    hint: S.get('gemini_hint'),
                    trailingLabel: _buildGeminiTierToggle(),
                  ),
                  _buildProviderCard(
                    provider: 'anthropic',
                    label: 'Anthropic',
                    keyCtrl: _anthropicKeyCtrl,
                  ),
                  _buildVertexAiCard(),
                  _buildProviderCard(
                    provider: 'custom',
                    label: S.get('custom_provider_label'),
                    keyCtrl: _customKeyCtrl,
                    baseUrlCtrl: _customBaseUrlCtrl,
                    modelCtrl: _customModelCtrl,
                    hint: S.get('custom_provider_hint'),
                  ),

                  const SizedBox(height: 24),

                  // ---- Debate Settings ----
                  _sectionHeader(S.get('debate_settings')),
                  const SizedBox(height: 8),
                  Card(
                    color: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                      side: const BorderSide(color: _cardBorder),
                    ),
                    elevation: 0,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        children: [
                          Row(
                            children: [
                              Expanded(
                                child: TextField(
                                  controller: _teamANameCtrl,
                                  decoration: InputDecoration(
                                    labelText: S.get('team_a_name_label'),
                                    filled: true,
                                    fillColor: _inputFill,
                                    border: const OutlineInputBorder(),
                                    enabledBorder: const OutlineInputBorder(
                                      borderSide: BorderSide(color: _inputBorder),
                                    ),
                                    isDense: true,
                                  ),
                                  onChanged: (_) => _autoSaveDebateSettings(),
                                ),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: TextField(
                                  controller: _teamBNameCtrl,
                                  decoration: InputDecoration(
                                    labelText: S.get('team_b_name_label'),
                                    filled: true,
                                    fillColor: _inputFill,
                                    border: const OutlineInputBorder(),
                                    enabledBorder: const OutlineInputBorder(
                                      borderSide: BorderSide(color: _inputBorder),
                                    ),
                                    isDense: true,
                                  ),
                                  onChanged: (_) => _autoSaveDebateSettings(),
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 8),
                          _numericField(S.get('min_rounds'), _minRoundsCtrl,
                              onChanged: _autoSaveDebateSettings),
                          _numericField(S.get('max_rounds'), _maxRoundsCtrl,
                              onChanged: _autoSaveDebateSettings),
                          _numericField(S.get('team_discussion_turns'),
                              _teamDiscussionTurnsCtrl,
                              onChanged: _autoSaveDebateSettings),
                          _numericField(S.get('judge_early_stop'),
                              _judgeEarlyStopCtrl,
                              onChanged: _autoSaveDebateSettings),
                          _numericField(S.get('chunk_size'), _chunkSizeCtrl,
                              onChanged: _autoSaveDebateSettings),
                          _numericField(S.get('chunk_overlap'), _chunkOverlapCtrl,
                              onChanged: _autoSaveDebateSettings),
                          _numericField(S.get('rag_top_k'), _ragTopKCtrl,
                              onChanged: _autoSaveDebateSettings),
                          _numericField(S.get('team_size'), _teamSizeCtrl,
                              onChanged: _autoSaveDebateSettings),
                          _numericField(S.get('judge_count'), _judgeCountCtrl,
                              onChanged: _autoSaveDebateSettings),
                          _numericField(S.get('max_review_more'), _maxReviewMoreCtrl,
                              onChanged: _autoSaveDebateSettings,
                              hint: S.get('max_review_more_hint')),
                          _numericField(S.get('accept_ratio'), _acceptRatioCtrl,
                              onChanged: _autoSaveDebateSettings,
                              hint: S.get('accept_ratio_hint')),
                          _numericField(S.get('review_more_ratio'), _reviewMoreRatioCtrl,
                              onChanged: _autoSaveDebateSettings,
                              hint: S.get('review_more_ratio_hint')),
                          const SizedBox(height: 8),
                          Row(
                            children: [
                              Text(
                                S.get('llm_temperature'),
                                style: const TextStyle(color: Colors.black87),
                              ),
                              Expanded(
                                child: Slider(
                                  value: _temperature,
                                  min: 0.0,
                                  max: 1.0,
                                  divisions: 20,
                                  activeColor: _accent,
                                  inactiveColor: _cardBorder,
                                  label: _temperature.toStringAsFixed(2),
                                  onChanged: (val) {
                                    setState(() => _temperature = val);
                                    _autoSaveDebateSettings();
                                  },
                                ),
                              ),
                              SizedBox(
                                width: 48,
                                child: Text(
                                  _temperature.toStringAsFixed(2),
                                  textAlign: TextAlign.center,
                                  style: const TextStyle(color: Colors.black87),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 32),

                  // Error display.
                  if (settings.error != null)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 16),
                      child: Text(
                        settings.error!,
                        style: const TextStyle(color: Colors.red),
                      ),
                    ),
                ],
              ),
            ),
      ),
    );
  }

  /// Build the Vertex AI card with service account JSON, project ID, location.
  Widget _buildVertexAiCard() {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      color: Colors.white,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: const BorderSide(color: _cardBorder),
      ),
      elevation: 0,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    const Text(
                      'Vertex AI (Google Cloud)',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: Colors.black87,
                      ),
                    ),
                    const SizedBox(width: 4),
                    IconButton(
                      icon: const Icon(Icons.help_outline, size: 16, color: _subtleText),
                      onPressed: () async {
                        final uri = Uri.parse(_pricingUrls['vertex_ai']!);
                        if (await canLaunchUrl(uri)) {
                          await launchUrl(uri, mode: LaunchMode.externalApplication);
                        }
                      },
                      tooltip: S.get('pricing_info'),
                      constraints: const BoxConstraints(),
                      padding: const EdgeInsets.all(4),
                    ),
                  ],
                ),
                _providerStatus('vertex_ai'),
              ],
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _vertexProjectIdCtrl,
              onChanged: (_) => _autoSaveProvider('vertex_ai'),
              style: const TextStyle(color: Colors.black87),
              decoration: _inputDecoration(S.get('project_id_label'), icon: Icons.folder),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _vertexLocationCtrl,
              onChanged: (_) => _autoSaveProvider('vertex_ai'),
              style: const TextStyle(color: Colors.black87),
              decoration: _inputDecoration(S.get('location_label'),
                  icon: Icons.location_on),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _vertexKeyCtrl,
              obscureText: _obscureKeys['vertex_ai'] ?? true,
              enableInteractiveSelection: true,
              enableSuggestions: false,
              autocorrect: false,
              onChanged: (_) {
                setState(() {});
                _autoSaveProvider('vertex_ai');
              },
              style: const TextStyle(color: Colors.black87),
              decoration:
                  _inputDecoration(S.get('api_key'), icon: Icons.key).copyWith(
                suffixIcon: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (_vertexKeyCtrl.text.isNotEmpty)
                      IconButton(
                        icon: Icon(
                          (_obscureKeys['vertex_ai'] ?? true)
                              ? Icons.visibility_off
                              : Icons.visibility,
                          size: 16,
                          color: _subtleText,
                        ),
                        onPressed: () => setState(() =>
                            _obscureKeys['vertex_ai'] =
                                !(_obscureKeys['vertex_ai'] ?? true)),
                        constraints: const BoxConstraints(),
                        padding: const EdgeInsets.all(8),
                      ),
                  ],
                ),
              ),
            ),
            // Model selection for Vertex AI with pricing
            const SizedBox(height: 12),
            Row(
              children: [
                Text(S.get('model_label'),
                    style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600,
                        color: _subtleText)),
                const SizedBox(width: 6),
                Text(S.get('per_1m_tokens'),
                    style: const TextStyle(fontSize: 10, color: Color(0xFF999999))),
              ],
            ),
            const SizedBox(height: 6),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              children: _presetModelsFor('vertex_ai').map((model) {
                final selected = _selectedModels['vertex_ai'] == model;
                final price = _presetModelPricing['vertex_ai']?[model] ?? '';
                return ChoiceChip(
                  label: Text('$model  $price', style: const TextStyle(fontSize: 11)),
                  selected: selected,
                  selectedColor: Colors.black,
                  backgroundColor: Colors.white,
                  labelStyle: TextStyle(
                    color: selected ? Colors.white : Colors.black87,
                    fontSize: 11,
                  ),
                  side: BorderSide(
                    color: selected ? Colors.black : _cardBorder,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(4),
                  ),
                  onSelected: (sel) {
                    setState(() => _selectedModels['vertex_ai'] = sel ? model : null);
                    _autoSaveProvider('vertex_ai');
                  },
                );
              }).toList(),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                OutlinedButton.icon(
                  onPressed: () => _testProvider('vertex_ai'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.black87,
                    side: const BorderSide(color: _cardBorder),
                  ),
                  icon: _testResults['vertex_ai'] == null &&
                          _testResults.containsKey('vertex_ai')
                      ? const SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(
                            strokeWidth: 2, color: _accent),
                        )
                      : const Icon(Icons.wifi_tethering, size: 16),
                  label: Text(S.get('test')),
                ),
                const SizedBox(width: 8),
                if (_testResults['vertex_ai'] == true)
                  Text(S.get('success'),
                      style: const TextStyle(color: Colors.green, fontSize: 12))
                else if (_testResults['vertex_ai'] == false)
                  Text(S.get('failed'),
                      style: const TextStyle(color: Colors.red, fontSize: 12)),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _sectionHeader(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 20,
        fontWeight: FontWeight.bold,
        letterSpacing: 0.3,
        color: Colors.black87,
      ),
    );
  }
}

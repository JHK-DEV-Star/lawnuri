import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../api/settings_api.dart';
import '../l10n/app_strings.dart';

/// State model for application settings.
class SettingsState {
  final List<Map<String, dynamic>> availableModels;
  final Map<String, dynamic> providers;
  final Map<String, dynamic> debateSettings;
  final Map<String, dynamic> legalApiSettings;
  final bool isLoading;
  final String? error;

  const SettingsState({
    this.availableModels = const [],
    this.providers = const {},
    this.debateSettings = const {},
    this.legalApiSettings = const {},
    this.isLoading = false,
    this.error,
  });

  /// Create a copy with optional field overrides.
  SettingsState copyWith({
    List<Map<String, dynamic>>? availableModels,
    Map<String, dynamic>? providers,
    Map<String, dynamic>? debateSettings,
    Map<String, dynamic>? legalApiSettings,
    bool? isLoading,
    String? error,
  }) {
    return SettingsState(
      availableModels: availableModels ?? this.availableModels,
      providers: providers ?? this.providers,
      debateSettings: debateSettings ?? this.debateSettings,
      legalApiSettings: legalApiSettings ?? this.legalApiSettings,
      isLoading: isLoading ?? this.isLoading,
      error: error,
    );
  }
}

/// StateNotifier that manages application settings.
class SettingsNotifier extends StateNotifier<SettingsState> {
  final SettingsApi _api = SettingsApi();

  SettingsNotifier() : super(const SettingsState());

  /// Load all settings from the backend.
  Future<void> loadAll() async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final results = await Future.wait([
        _api.getAvailableModels(),
        _api.getProviders(),
        _api.getDebateSettings(),
        _api.getLegalApiSettings(),
      ]);

      final debateSettings = results[2] as Map<String, dynamic>;

      final lang = debateSettings['language'] as String? ?? 'ko';
      S.setLanguage(lang);

      state = state.copyWith(
        availableModels: results[0] as List<Map<String, dynamic>>,
        providers: results[1] as Map<String, dynamic>,
        debateSettings: debateSettings,
        legalApiSettings: results[3] as Map<String, dynamic>,
        isLoading: false,
      );
    } catch (e) {
      state = state.copyWith(error: e.toString(), isLoading: false);
    }
  }

  /// Load available models (with caching -- skips if already loaded).
  Future<void> loadModels({bool force = false}) async {
    if (!force && state.availableModels.isNotEmpty) return;
    try {
      final models = await _api.getAvailableModels();
      state = state.copyWith(availableModels: models);
    } catch (e) {
      state = state.copyWith(error: e.toString());
    }
  }

  /// Load LLM provider status.
  Future<void> loadProviders() async {
    try {
      final providers = await _api.getProviders();
      state = state.copyWith(providers: providers);
    } catch (e) {
      state = state.copyWith(error: e.toString());
    }
  }

  /// Update an LLM provider configuration.
  Future<void> updateProvider(
    String provider,
    String apiKey, {
    String? baseUrl,
    String? model,
    bool enabled = true,
  }) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      await _api.updateProvider(
        provider,
        apiKey,
        baseUrl: baseUrl,
        model: model,
        enabled: enabled,
      );
      await loadProviders();
      await loadModels(force: true);
      state = state.copyWith(isLoading: false);
    } catch (e) {
      state = state.copyWith(error: e.toString(), isLoading: false);
    }
  }

  /// Test connectivity to a specific LLM provider.
  Future<Map<String, dynamic>> testProvider(String provider) async {
    try {
      return await _api.testProvider(provider);
    } catch (e) {
      state = state.copyWith(error: e.toString());
      return {'success': false, 'error': e.toString()};
    }
  }

  /// Load debate settings.
  Future<void> loadDebateSettings() async {
    try {
      final settings = await _api.getDebateSettings();
      state = state.copyWith(debateSettings: settings);
    } catch (e) {
      state = state.copyWith(error: e.toString());
    }
  }

  /// Update debate settings.
  Future<void> updateDebateSettings(Map<String, dynamic> settings) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      await _api.updateDebateSettings(settings);
      await loadDebateSettings();
      state = state.copyWith(isLoading: false);
    } catch (e) {
      state = state.copyWith(error: e.toString(), isLoading: false);
    }
  }

  /// Clear the error state.
  void clearError() {
    state = state.copyWith(error: null);
  }
}

/// Global provider for settings state.
final settingsProvider = StateNotifierProvider<SettingsNotifier, SettingsState>(
  (ref) => SettingsNotifier(),
);

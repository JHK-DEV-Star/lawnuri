import 'package:dio/dio.dart';
import 'api_client.dart';

/// API client for application settings and LLM provider management.
class SettingsApi {
  final Dio _dio = ApiClient().dio;

  /// Get available debate presets.
  Future<Map<String, dynamic>> getPresets() async {
    final response = await _dio.get('/api/settings/presets');
    return Map<String, dynamic>.from(
        (response.data as Map<String, dynamic>)['presets'] as Map);
  }

  /// Get configured LLM providers and their status.
  Future<Map<String, dynamic>> getProviders() async {
    final response = await _dio.get('/api/settings/providers');
    return Map<String, dynamic>.from(
        (response.data as Map<String, dynamic>)['providers'] as Map);
  }

  /// Update an LLM provider's configuration.
  Future<void> updateProvider(
    String provider,
    String apiKey, {
    String? baseUrl,
    String? model,
    bool enabled = true,
  }) async {
    final data = <String, dynamic>{
      'api_key': apiKey,
      'enabled': enabled,
    };
    if (baseUrl != null) data['base_url'] = baseUrl;
    if (model != null) data['model'] = model;

    await _dio.put('/api/settings/providers/$provider', data: data);
  }

  /// Update a provider with a raw data map (for Vertex AI service account etc.).
  Future<void> updateProviderRaw(
      String provider, Map<String, dynamic> data) async {
    await _dio.put('/api/settings/providers/$provider', data: data);
  }

  /// Test connectivity to an LLM provider.
  Future<Map<String, dynamic>> testProvider(String provider) async {
    final response = await _dio.post('/api/settings/providers/$provider/test');
    return response.data as Map<String, dynamic>;
  }

  /// Test connectivity to the legal API (국가법령정보센터).
  Future<Map<String, dynamic>> testLegalApi() async {
    final response = await _dio.post('/api/settings/legal-api/test');
    return response.data as Map<String, dynamic>;
  }

  /// Get the list of all available LLM models across providers.
  Future<List<Map<String, dynamic>>> getAvailableModels() async {
    final response = await _dio.get('/api/settings/models');
    final list = (response.data as Map<String, dynamic>)['models'] as List;
    return list.map((m) => Map<String, dynamic>.from(m as Map)).toList();
  }

  /// Get current debate settings (rounds, timing, etc.).
  Future<Map<String, dynamic>> getDebateSettings() async {
    final response = await _dio.get('/api/settings/debate');
    return Map<String, dynamic>.from(
        (response.data as Map<String, dynamic>)['debate'] as Map);
  }

  /// Update debate settings.
  Future<void> updateDebateSettings(Map<String, dynamic> settings) async {
    await _dio.put('/api/settings/debate', data: settings);
  }

  /// Get legal API settings (OC key, max API calls).
  Future<Map<String, dynamic>> getLegalApiSettings() async {
    final response = await _dio.get('/api/settings/legal-api');
    return Map<String, dynamic>.from(
        (response.data as Map<String, dynamic>)['legal_api'] as Map);
  }

  /// Update legal API settings (OC key, max API calls).
  Future<Map<String, dynamic>> updateLegalApiSettings(
      Map<String, dynamic> settings) async {
    final response =
        await _dio.put('/api/settings/legal-api', data: settings);
    return response.data as Map<String, dynamic>;
  }

  /// Get all application settings at once.
  Future<Map<String, dynamic>> getAllSettings() async {
    final response = await _dio.get('/api/settings');
    return Map<String, dynamic>.from(
        (response.data as Map<String, dynamic>)['settings'] as Map);
  }
}

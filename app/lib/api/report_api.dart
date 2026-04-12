import 'package:dio/dio.dart';
import 'api_client.dart';

/// API client for debate report generation and download.
class ReportApi {
  final Dio _dio = ApiClient().dio;

  /// Get the debate report data (markdown content, statistics, etc.).
  Future<Map<String, dynamic>> getReport(String debateId) async {
    final response = await _dio.get('/api/report/$debateId');
    final data = response.data as Map<String, dynamic>;
    // Backend wraps report in {"report": {...}} — unwrap if present
    return data['report'] as Map<String, dynamic>? ?? data;
  }

  /// Regenerate the report (clear cache and generate fresh).
  Future<Map<String, dynamic>> regenerateReport(String debateId) async {
    final response = await _dio.post('/api/report/$debateId/regenerate');
    final data = response.data as Map<String, dynamic>;
    return data['report'] as Map<String, dynamic>? ?? data;
  }

  /// Download the debate report as a file to the specified path.
  Future<void> downloadReport(String debateId, String savePath) async {
    await _dio.download(
      '/api/report/$debateId/download',
      savePath,
      options: Options(
        responseType: ResponseType.bytes,
      ),
    );
  }
}

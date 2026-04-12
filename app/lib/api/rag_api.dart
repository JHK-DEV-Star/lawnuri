import 'dart:io';
import 'package:dio/dio.dart';
import 'api_client.dart';

/// API client for RAG (Retrieval-Augmented Generation) and document management.
class RagApi {
  final Dio _dio = ApiClient().dio;

  /// Upload a document to the debate's knowledge pool.
  Future<Map<String, dynamic>> uploadDocument(
    String debateId,
    String pool,
    File file,
  ) async {
    final formData = FormData.fromMap({
      'file': await MultipartFile.fromFile(
        file.path,
        filename: file.path.split(Platform.pathSeparator).last,
      ),
      'pool': pool,
      'debate_id': debateId,
    });

    final response = await _dio.post(
      '/api/rag/upload',
      data: formData,
      options: Options(contentType: 'multipart/form-data'),
    );
    return response.data as Map<String, dynamic>;
  }

  /// Start indexing documents for a debate.
  /// Returns a task ID that can be used to check indexing status.
  Future<String> startIndexing(String debateId, {String? pool}) async {
    final data = <String, dynamic>{};
    if (pool != null) data['pool'] = pool;

    data['debate_id'] = debateId;
    final response = await _dio.post(
      '/api/rag/index',
      data: data,
    );
    return (response.data as Map<String, dynamic>)['task_id'] as String;
  }

  /// Check the status of an indexing task.
  Future<Map<String, dynamic>> getIndexingStatus(String taskId) async {
    final response = await _dio.get('/api/rag/index/status/$taskId');
    return response.data as Map<String, dynamic>;
  }

  /// Search the debate's knowledge base.
  Future<List<dynamic>> search(
    String debateId,
    String query, {
    String pool = 'common',
    String searchType = 'both',
  }) async {
    final response = await _dio.post(
      '/api/rag/search',
      data: {
        'query': query,
        'debate_id': debateId,
        'pool': pool,
        'search_type': searchType,
      },
    );
    final data = response.data;
    if (data is Map) {
      return (data['results'] as List<dynamic>?) ?? [];
    }
    return data as List<dynamic>;
  }
}

/// API configuration constants for the LawNuri backend.
class ApiConfig {
  /// Base URL of the backend API server.
  static const String baseUrl = 'http://localhost:8000';

  /// Default timeout for regular API requests.
  static const Duration timeout = Duration(seconds: 30);

  /// Extended timeout for long-running operations (debate rounds, analysis).
  static const Duration longTimeout = Duration(seconds: 120);
}

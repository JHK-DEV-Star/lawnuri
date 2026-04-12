import 'package:dio/dio.dart';
import '../config/api_config.dart';

/// Singleton Dio HTTP client with retry logic, timeout, and logging.
class ApiClient {
  static ApiClient? _instance;
  late final Dio dio;

  /// Maximum number of retry attempts for failed requests.
  static const int _maxRetries = 3;

  /// Delay between retry attempts.
  static const Duration _retryDelay = Duration(seconds: 1);

  ApiClient._internal() {
    dio = Dio(
      BaseOptions(
        baseUrl: ApiConfig.baseUrl,
        connectTimeout: ApiConfig.timeout,
        receiveTimeout: ApiConfig.longTimeout,
        sendTimeout: ApiConfig.timeout,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
      ),
    );

    dio.interceptors.add(
      LogInterceptor(
        request: true,
        requestHeader: false,
        requestBody: true,
        responseHeader: false,
        responseBody: true,
        error: true,
        logPrint: (object) => print('[API] $object'),
      ),
    );

    dio.interceptors.add(_RetryInterceptor(dio: dio));
  }

  /// Get the singleton ApiClient instance.
  factory ApiClient() {
    _instance ??= ApiClient._internal();
    return _instance!;
  }

  /// Get the underlying Dio instance.
  Dio get client => dio;
}

/// Interceptor that retries failed requests on server errors or timeouts.
class _RetryInterceptor extends Interceptor {
  final Dio dio;

  _RetryInterceptor({required this.dio});

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) async {
    // Only retry on connection errors, timeouts, and 5xx server errors.
    if (_shouldRetry(err)) {
      final retryCount = err.requestOptions.extra['retryCount'] ?? 0;

      if (retryCount < ApiClient._maxRetries) {
        err.requestOptions.extra['retryCount'] = retryCount + 1;

        // Wait before retrying with exponential backoff.
        final delay = ApiClient._retryDelay * (retryCount + 1);
        await Future.delayed(delay);

        try {
          final response = await dio.fetch(err.requestOptions);
          handler.resolve(response);
          return;
        } on DioException catch (e) {
          handler.next(e);
          return;
        }
      }
    }

    handler.next(err);
  }

  /// Determine whether a request should be retried based on the error type.
  bool _shouldRetry(DioException err) {
    if (err.type == DioExceptionType.connectionTimeout ||
        err.type == DioExceptionType.receiveTimeout ||
        err.type == DioExceptionType.sendTimeout ||
        err.type == DioExceptionType.connectionError) {
      return true;
    }

    // Retry on 5xx server errors.
    final statusCode = err.response?.statusCode;
    if (statusCode != null && statusCode >= 500) {
      return true;
    }

    return false;
  }
}

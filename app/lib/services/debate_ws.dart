import 'dart:async';
import 'dart:convert';

import 'package:web_socket_channel/web_socket_channel.dart';

import '../config/api_config.dart';

/// Real-time WebSocket client for debate event streaming.
///
/// Connects to the backend's `/api/debate/{debateId}/ws` endpoint and
/// exposes an event stream that the provider can subscribe to.
/// Automatically reconnects on disconnection with exponential backoff.
class DebateWebSocket {
  final String debateId;
  WebSocketChannel? _channel;
  StreamController<Map<String, dynamic>>? _controller;
  Timer? _reconnectTimer;
  int _reconnectAttempts = 0;
  static const int _maxReconnectAttempts = 5;
  bool _disposed = false;

  DebateWebSocket({required this.debateId}) {
    _controller = StreamController<Map<String, dynamic>>.broadcast();
  }

  /// Stream of decoded JSON events from the server.
  /// Returns an empty stream if the controller has been disposed.
  Stream<Map<String, dynamic>> get events =>
      _controller?.stream ?? const Stream.empty();

  /// Connect to the WebSocket endpoint.
  void connect() {
    if (_disposed) return;

    final wsUrl = ApiConfig.baseUrl
        .replaceFirst('http://', 'ws://')
        .replaceFirst('https://', 'wss://');
    final uri = Uri.parse('$wsUrl/api/debate/$debateId/ws');

    try {
      _channel = WebSocketChannel.connect(uri);

      _channel!.stream.listen(
        (data) {
          if (_disposed) return;
          _reconnectAttempts = 0;
          try {
            final decoded =
                jsonDecode(data as String) as Map<String, dynamic>;
            _controller?.add(decoded);
          } catch (e) {
            assert(() {
              // ignore: avoid_print
              print('[DebateWebSocket] Failed to decode WS message: $e');
              return true;
            }());
          }
        },
        onError: (error) {
          if (!_disposed) _scheduleReconnect();
        },
        onDone: () {
          if (!_disposed) _scheduleReconnect();
        },
      );
    } catch (_) {
      _scheduleReconnect();
    }
  }

  void _scheduleReconnect() {
    if (_disposed) return;
    if (_reconnectAttempts >= _maxReconnectAttempts) {
      _controller?.close();
      return;
    }
    _reconnectAttempts++;
    // Exponential backoff: 1s, 2s, 4s, 8s...
    final delay = Duration(seconds: 1 << _reconnectAttempts);
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(delay, connect);
  }

  /// Send a keep-alive ping (or any text message) to the server.
  void ping() {
    try {
      _channel?.sink.add('ping');
    } catch (_) {}
  }

  /// Close the connection and release resources.
  void dispose() {
    _disposed = true;
    _reconnectTimer?.cancel();
    _channel?.sink.close();
    _controller?.close();
    _controller = null;
  }
}

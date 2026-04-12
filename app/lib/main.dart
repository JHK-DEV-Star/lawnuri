import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import 'screens/home_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/debate_screen.dart';
import 'screens/report_screen.dart';

/// Managed backend server process (null when running in dev mode).
Process? _serverProcess;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // In bundled distribution mode, auto-launch the backend server.
  await _startBackendServer();

  runApp(
    const ProviderScope(
      child: LawNuriApp(),
    ),
  );
}

Future<void> _startBackendServer() async {
  final exePath = Platform.resolvedExecutable;
  final appDir = File(exePath).parent.path;
  final serverExe = '$appDir/server/lawnuri_server.exe';

  if (!File(serverExe).existsSync()) {
    debugPrint('[LawNuri] Bundled server not found, skipping auto-launch.');
    return;
  }

  debugPrint('[LawNuri] Starting backend server: $serverExe');
  try {
    _serverProcess = await Process.start(
      serverExe,
      [],
      workingDirectory: '$appDir/server',
      mode: ProcessStartMode.detached,
      // Pass our PID so backend's watchdog can self-exit when we die.
      environment: {'LAWNURI_PARENT_PID': pid.toString()},
    );
    debugPrint('[LawNuri] Backend server started (PID: ${_serverProcess!.pid})');

    await Future.delayed(const Duration(seconds: 2));
  } catch (e) {
    debugPrint('[LawNuri] Failed to start backend server: $e');
  }
}

class LawNuriApp extends StatefulWidget {
  const LawNuriApp({super.key});

  @override
  State<LawNuriApp> createState() => _LawNuriAppState();
}

class _LawNuriAppState extends State<LawNuriApp> with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _killBackendServer();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.detached) {
      _killBackendServer();
    }
  }

  void _killBackendServer() {
    if (_serverProcess == null) return;
    final p = _serverProcess!.pid;
    debugPrint('[LawNuri] Killing backend server (PID: $p)');
    try {
      if (Platform.isWindows) {
        // Kill the bootloader AND its python child (full process tree).
        Process.runSync('taskkill', ['/F', '/T', '/PID', p.toString()]);
      } else {
        Process.killPid(p, ProcessSignal.sigterm);
      }
    } catch (e) {
      debugPrint('[LawNuri] kill failed: $e');
    }
    _serverProcess = null;
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'Law-Nuri',
      debugShowCheckedModeBanner: false,
      theme: _buildLightTheme(),
      routerConfig: _router,
    );
  }

  /// MiroFish-inspired light theme: white bg, black primary, orange accent.
  static ThemeData _buildLightTheme() {
    const accent = Color(0xFFFF4500);
    const border = Color(0xFFE5E5E5);
    const inputBg = Color(0xFFFAFAFA);
    const textSecondary = Color(0xFF666666);

    return ThemeData(
      brightness: Brightness.light,
      colorScheme: const ColorScheme.light(
        primary: Colors.black,
        secondary: accent,
        surface: Colors.white,
        onPrimary: Colors.white,
        onSecondary: Colors.white,
        onSurface: Colors.black,
        error: Color(0xFFF44336),
      ),
      scaffoldBackgroundColor: Colors.white,
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
        elevation: 0,
        centerTitle: false,
        scrolledUnderElevation: 0,
        titleTextStyle: TextStyle(
          color: Colors.black,
          fontSize: 18,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.5,
        ),
      ),
      cardTheme: CardThemeData(
        color: Colors.white,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(4),
          side: const BorderSide(color: border),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.black,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(4),
          ),
          textStyle: const TextStyle(
            fontWeight: FontWeight.w700,
            fontSize: 15,
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: Colors.black,
          side: const BorderSide(color: Colors.black),
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(4),
          ),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: Colors.black,
          foregroundColor: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(4),
          ),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: inputBg,
        hintStyle: const TextStyle(color: textSecondary, fontSize: 14),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(4),
          borderSide: const BorderSide(color: Color(0xFFDDDDDD)),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(4),
          borderSide: const BorderSide(color: Color(0xFFDDDDDD)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(4),
          borderSide: const BorderSide(color: Colors.black, width: 1.5),
        ),
      ),
      dividerColor: const Color(0xFFEAEAEA),
      chipTheme: ChipThemeData(
        backgroundColor: Colors.white,
        selectedColor: Colors.black,
        labelStyle: const TextStyle(color: Colors.black, fontSize: 13),
        secondaryLabelStyle: const TextStyle(color: Colors.white, fontSize: 13),
        side: const BorderSide(color: border),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
      ),
      useMaterial3: true,
    );
  }

  /// Application router configuration.
  static final GoRouter _router = GoRouter(
    initialLocation: '/',
    routes: [
      GoRoute(
        path: '/',
        name: 'home',
        builder: (context, state) => const HomeScreen(),
      ),
      GoRoute(
        path: '/settings',
        name: 'settings',
        builder: (context, state) => const SettingsScreen(),
      ),
      GoRoute(
        path: '/debate/:id',
        name: 'debate',
        builder: (context, state) {
          final debateId = state.pathParameters['id']!;
          return DebateScreen(debateId: debateId);
        },
      ),
      GoRoute(
        path: '/verdict/:id',
        name: 'verdict',
        redirect: (context, state) => '/debate/${state.pathParameters['id']}',
      ),
      GoRoute(
        path: '/report/:id',
        name: 'report',
        builder: (context, state) {
          final debateId = state.pathParameters['id']!;
          return ReportScreen(debateId: debateId);
        },
      ),
    ],
  );
}

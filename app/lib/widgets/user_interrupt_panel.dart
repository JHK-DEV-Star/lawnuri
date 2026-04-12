import 'package:flutter/material.dart';

import '../l10n/app_strings.dart';

/// Compact horizontal panel for user intervention during a live debate.
///
/// Allows the user to select a target team, type a hint or comment,
/// attach a file, and send the intervention to the debate engine.
class UserInterruptPanel extends StatefulWidget {
  /// Callback when the user sends an intervention.
  /// Parameters: target team, content text, intervention type.
  final Function(String targetTeam, String content, String type) onSend;

  /// Optional callback for file upload.
  /// Parameters: target team, file reference (platform-dependent).
  final Function(String targetTeam, dynamic file)? onFileUpload;

  /// Optional custom team names for display.
  final String? teamAName;
  final String? teamBName;

  const UserInterruptPanel({
    super.key,
    required this.onSend,
    this.onFileUpload,
    this.teamAName,
    this.teamBName,
  });

  @override
  State<UserInterruptPanel> createState() => _UserInterruptPanelState();
}

class _UserInterruptPanelState extends State<UserInterruptPanel> {
  final TextEditingController _textController = TextEditingController();

  /// Currently selected target team.
  String _targetTeam = 'team_a';

  /// Currently selected intervention type.
  String _interventionType = 'hint';

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

  /// Send the intervention and clear the input.
  void _handleSend() {
    final text = _textController.text.trim();
    if (text.isEmpty) return;

    widget.onSend(_targetTeam, text, _interventionType);
    _textController.clear();
  }

  /// Trigger the file upload callback.
  void _handleFileUpload() {
    if (widget.onFileUpload != null) {
      widget.onFileUpload!(_targetTeam, null);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: const BoxDecoration(
        color: Color(0xFFFAFAFA),
        border: Border(
          top: BorderSide(
            color: Color(0xFFE5E5E5),
          ),
        ),
      ),
      child: Row(
        children: [
          _buildTeamSelector(),
          const SizedBox(width: 10),

          _buildTypeDropdown(),
          const SizedBox(width: 10),

          Expanded(
            child: TextField(
              controller: _textController,
              decoration: InputDecoration(
                hintText: _hintText,
                hintStyle: const TextStyle(color: Color(0xFF999999)),
                isDense: true,
                contentPadding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                border: const OutlineInputBorder(
                  borderSide: BorderSide(color: Color(0xFFE5E5E5)),
                ),
                enabledBorder: const OutlineInputBorder(
                  borderSide: BorderSide(color: Color(0xFFE5E5E5)),
                ),
              ),
              style: const TextStyle(fontSize: 13, color: Colors.black87),
              onSubmitted: (_) => _handleSend(),
            ),
          ),
          const SizedBox(width: 8),

          if (widget.onFileUpload != null)
            IconButton(
              icon: const Icon(Icons.attach_file, size: 20, color: Color(0xFF666666)),
              tooltip: S.get('attach_evidence_file'),
              onPressed: _handleFileUpload,
              style: IconButton.styleFrom(
                padding: const EdgeInsets.all(8),
              ),
            ),
          const SizedBox(width: 4),

          ElevatedButton.icon(
            onPressed: _handleSend,
            icon: const Icon(Icons.send, size: 16),
            label: Text(S.get('send')),
            style: ElevatedButton.styleFrom(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            ),
          ),
        ],
      ),
    );
  }

  /// Build the segmented button for team selection.
  Widget _buildTeamSelector() {
    return SegmentedButton<String>(
      segments: [
        ButtonSegment(
          value: 'team_a',
          label: Text(S.teamName('team_a', teamAName: widget.teamAName, teamBName: widget.teamBName), style: const TextStyle(fontSize: 12)),
          icon: const Icon(Icons.groups, size: 16),
        ),
        ButtonSegment(
          value: 'team_b',
          label: Text(S.teamName('team_b', teamAName: widget.teamAName, teamBName: widget.teamBName), style: const TextStyle(fontSize: 12)),
          icon: const Icon(Icons.groups, size: 16),
        ),
      ],
      selected: {_targetTeam},
      onSelectionChanged: (selection) {
        setState(() => _targetTeam = selection.first);
      },
      style: ButtonStyle(
        visualDensity: VisualDensity.compact,
        tapTargetSize: MaterialTapTargetSize.shrinkWrap,
        padding: WidgetStateProperty.all(
          const EdgeInsets.symmetric(horizontal: 8),
        ),
      ),
    );
  }

  /// Build the dropdown for intervention type selection.
  Widget _buildTypeDropdown() {
    return DropdownButton<String>(
      value: _interventionType,
      underline: const SizedBox.shrink(),
      isDense: true,
      style: const TextStyle(fontSize: 12, color: Colors.black87),
      items: [
        DropdownMenuItem(value: 'hint', child: Text(S.get('hint'))),
        DropdownMenuItem(value: 'evidence', child: Text(S.get('evidence_label'))),
        DropdownMenuItem(value: 'objection', child: Text(S.get('objection'))),
        DropdownMenuItem(value: 'question', child: Text(S.get('question_label'))),
      ],
      onChanged: (val) {
        if (val != null) setState(() => _interventionType = val);
      },
    );
  }

  /// Dynamic hint text based on the selected intervention type.
  String get _hintText {
    switch (_interventionType) {
      case 'hint':
        return S.get('enter_hint_for_team');
      case 'evidence':
        return S.get('describe_evidence');
      case 'objection':
        return S.get('state_your_objection');
      case 'question':
        return S.get('ask_question_team');
      default:
        return S.get('type_your_message');
    }
  }
}

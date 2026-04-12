import 'package:flutter/material.dart';

import '../models/agent.dart';

/// Card widget displaying a single agent's profile with team color,
/// role badge, attribute tags, optional LLM override dropdown, and edit button.
class AgentProfileCard extends StatelessWidget {
  /// The agent profile data to display.
  final AgentProfile agent;

  /// Available LLM models for the override dropdown.
  /// Each map should contain "id" and "name" keys.
  final List<Map<String, dynamic>>? availableModels;

  /// Callback invoked with updated fields when the user edits the agent.
  final Function(Map<String, dynamic>)? onEdit;

  /// Color representing the agent's team.
  final Color teamColor;

  const AgentProfileCard({
    super.key,
    required this.agent,
    this.availableModels,
    this.onEdit,
    required this.teamColor,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      clipBehavior: Clip.antiAlias,
      color: Colors.white,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: const BorderSide(color: Color(0xFFE5E5E5)),
      ),
      child: IntrinsicHeight(
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(width: 4, color: teamColor),

            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildNameRow(context),
                    const SizedBox(height: 8),

                    _buildAttributeTags(),

                    if (availableModels != null &&
                        availableModels!.isNotEmpty) ...[
                      const SizedBox(height: 8),
                      _buildModelDropdown(context),
                    ],
                  ],
                ),
              ),
            ),

            if (onEdit != null)
              Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  IconButton(
                    icon: const Icon(Icons.edit_outlined, size: 20, color: Colors.black87),
                    tooltip: 'Edit agent',
                    onPressed: () => _showEditDialog(context),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }

  /// Build the header row with agent name and role badge.
  Widget _buildNameRow(BuildContext context) {
    return Row(
      children: [
        CircleAvatar(
          radius: 16,
          backgroundColor: teamColor.withValues(alpha: 0.2),
          child: Text(
            agent.name.isNotEmpty ? agent.name[0].toUpperCase() : '?',
            style: TextStyle(
              fontWeight: FontWeight.bold,
              color: teamColor,
              fontSize: 14,
            ),
          ),
        ),
        const SizedBox(width: 10),

        Expanded(
          child: Text(
            agent.name,
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 15,
              color: Colors.black87,
            ),
            overflow: TextOverflow.ellipsis,
          ),
        ),

        Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
          decoration: BoxDecoration(
            color: teamColor.withValues(alpha: 0.12),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: teamColor.withValues(alpha: 0.3)),
          ),
          child: Text(
            agent.role,
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: teamColor,
            ),
          ),
        ),
      ],
    );
  }

  /// Build the row of attribute chips for specialty, personality, debate style.
  Widget _buildAttributeTags() {
    final tags = <_TagData>[
      _TagData(label: agent.specialty, icon: Icons.school_outlined),
      _TagData(label: agent.personality, icon: Icons.psychology_outlined),
      _TagData(label: agent.debateStyle, icon: Icons.record_voice_over_outlined),
    ];

    return Wrap(
      spacing: 6,
      runSpacing: 4,
      children: tags.map((tag) {
        return Chip(
          avatar: Icon(tag.icon, size: 14, color: teamColor),
          label: Text(
            tag.label,
            style: const TextStyle(fontSize: 11, color: Colors.black87),
          ),
          materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
          visualDensity: VisualDensity.compact,
          padding: EdgeInsets.zero,
          side: BorderSide(color: teamColor.withValues(alpha: 0.2)),
          backgroundColor: Colors.white,
        );
      }).toList(),
    );
  }

  /// Build the LLM model override dropdown.
  Widget _buildModelDropdown(BuildContext context) {
    return Row(
      children: [
        const Icon(Icons.smart_toy_outlined, size: 16, color: Color(0xFF666666)),
        const SizedBox(width: 6),
        const Text(
          'LLM:',
          style: TextStyle(fontSize: 12, color: Color(0xFF666666)),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: DropdownButtonFormField<String>(
            value: agent.llmOverride,
            isDense: true,
            isExpanded: true,
            decoration: const InputDecoration(
              isDense: true,
              contentPadding:
                  EdgeInsets.symmetric(horizontal: 8, vertical: 6),
              border: OutlineInputBorder(),
            ),
            style: const TextStyle(fontSize: 12, color: Colors.black87),
            items: [
              const DropdownMenuItem(
                value: null,
                child: Text('Default', style: TextStyle(fontSize: 12, color: Colors.black87)),
              ),
              ...availableModels!.map((m) {
                final id = m['id'] as String? ?? '';
                final name = m['name'] as String? ?? id;
                return DropdownMenuItem(
                  value: id,
                  child: Text(
                    name,
                    style: const TextStyle(fontSize: 12, color: Colors.black87),
                    overflow: TextOverflow.ellipsis,
                  ),
                );
              }),
            ],
            onChanged: (val) {
              onEdit?.call({'llm_override': val});
            },
          ),
        ),
      ],
    );
  }

  /// Show a dialog for editing all agent fields.
  void _showEditDialog(BuildContext context) {
    final nameCtrl = TextEditingController(text: agent.name);
    final roleCtrl = TextEditingController(text: agent.role);
    final specialtyCtrl = TextEditingController(text: agent.specialty);
    final personalityCtrl = TextEditingController(text: agent.personality);
    final styleCtrl = TextEditingController(text: agent.debateStyle);
    final backgroundCtrl = TextEditingController(text: agent.background);

    showDialog(
      context: context,
      builder: (ctx) {
        return AlertDialog(
          title: Text('Edit ${agent.name}'),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: nameCtrl,
                  decoration: const InputDecoration(labelText: 'Name'),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: roleCtrl,
                  decoration: const InputDecoration(labelText: 'Role'),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: specialtyCtrl,
                  decoration: const InputDecoration(labelText: 'Specialty'),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: personalityCtrl,
                  decoration: const InputDecoration(labelText: 'Personality'),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: styleCtrl,
                  decoration: const InputDecoration(labelText: 'Debate Style'),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: backgroundCtrl,
                  decoration: const InputDecoration(labelText: 'Background'),
                  maxLines: 3,
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () {
                final updates = <String, dynamic>{
                  'name': nameCtrl.text,
                  'role': roleCtrl.text,
                  'specialty': specialtyCtrl.text,
                  'personality': personalityCtrl.text,
                  'debate_style': styleCtrl.text,
                  'background': backgroundCtrl.text,
                };
                Navigator.pop(ctx);
                onEdit?.call(updates);
              },
              child: const Text('Save'),
            ),
          ],
        );
      },
    ).then((_) {
      nameCtrl.dispose();
      roleCtrl.dispose();
      specialtyCtrl.dispose();
      personalityCtrl.dispose();
      styleCtrl.dispose();
      backgroundCtrl.dispose();
    });
  }
}

/// Simple data holder for attribute tag display.
class _TagData {
  final String label;
  final IconData icon;

  _TagData({required this.label, required this.icon});
}

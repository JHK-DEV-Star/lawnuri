import 'package:flutter/material.dart';

import '../models/agent.dart';
import 'agent_profile_card.dart';

/// Expandable panel for configuring debate agents.
///
/// Groups agents into Team A, Team B, and Judges sections with colored
/// headers, individual profile cards, and a regeneration button.
class AgentConfigPanel extends StatelessWidget {
  /// List of all agent profiles to display.
  final List<AgentProfile> agents;

  /// Available LLM models for per-agent override dropdowns.
  final List<Map<String, dynamic>> availableModels;

  /// Callback when an agent is updated. Receives agent ID and a map of changes.
  final Function(String agentId, Map<String, dynamic> updates) onAgentUpdate;

  /// Callback to regenerate all agents.
  final VoidCallback onGenerateAgents;

  /// Whether an async operation (e.g. agent generation) is in progress.
  final bool isLoading;

  const AgentConfigPanel({
    super.key,
    required this.agents,
    required this.availableModels,
    required this.onAgentUpdate,
    required this.onGenerateAgents,
    this.isLoading = false,
  });

  @override
  Widget build(BuildContext context) {
    final teamA = agents.where((a) => a.team == 'team_a').toList();
    final teamB = agents.where((a) => a.team == 'team_b').toList();
    final judges =
        agents.where((a) => a.team != 'team_a' && a.team != 'team_b').toList();

    return Card(
      color: Colors.white,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: const BorderSide(color: Color(0xFFE5E5E5)),
      ),
      child: ExpansionTile(
        leading: const Icon(Icons.people, color: Colors.black87),
        title: Text(
          'Agent Settings (${agents.length} agents)',
          style: const TextStyle(color: Colors.black87),
        ),
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                if (teamA.isNotEmpty) ...[
                  _buildSectionHeader(
                    context,
                    title: 'Team A',
                    color: Colors.blue,
                    count: teamA.length,
                  ),
                  const SizedBox(height: 8),
                  ...teamA.map((agent) => AgentProfileCard(
                        agent: agent,
                        availableModels: availableModels,
                        teamColor: Colors.blue,
                        onEdit: (updates) =>
                            onAgentUpdate(agent.agentId, updates),
                      )),
                  const SizedBox(height: 12),
                ],

                if (teamB.isNotEmpty) ...[
                  _buildSectionHeader(
                    context,
                    title: 'Team B',
                    color: Colors.red,
                    count: teamB.length,
                  ),
                  const SizedBox(height: 8),
                  ...teamB.map((agent) => AgentProfileCard(
                        agent: agent,
                        availableModels: availableModels,
                        teamColor: Colors.red,
                        onEdit: (updates) =>
                            onAgentUpdate(agent.agentId, updates),
                      )),
                  const SizedBox(height: 12),
                ],

                if (judges.isNotEmpty) ...[
                  _buildSectionHeader(
                    context,
                    title: 'Judges',
                    color: Colors.amber,
                    count: judges.length,
                  ),
                  const SizedBox(height: 8),
                  ...judges.map((agent) => AgentProfileCard(
                        agent: agent,
                        availableModels: availableModels,
                        teamColor: Colors.amber,
                        onEdit: (updates) =>
                            onAgentUpdate(agent.agentId, updates),
                      )),
                  const SizedBox(height: 12),
                ],

                if (agents.isEmpty)
                  Container(
                    padding: const EdgeInsets.all(24),
                    alignment: Alignment.center,
                    child: const Text(
                      'No agents generated yet. Click "Regenerate" to create agents.',
                      style: TextStyle(color: Color(0xFF999999)),
                    ),
                  ),

                const SizedBox(height: 8),

                ElevatedButton.icon(
                  onPressed: isLoading ? null : onGenerateAgents,
                  icon: isLoading
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.auto_awesome),
                  label: Text(isLoading ? 'Generating...' : 'Regenerate'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Build a colored section header for a team group.
  Widget _buildSectionHeader(
    BuildContext context, {
    required String title,
    required Color color,
    required int count,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.25)),
      ),
      child: Row(
        children: [
          Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(
              color: color,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 8),
          Text(
            title,
            style: TextStyle(
              fontWeight: FontWeight.w600,
              color: color,
              fontSize: 14,
            ),
          ),
          const Spacer(),
          Text(
            '$count agents',
            style: TextStyle(
              fontSize: 12,
              color: color.withValues(alpha: 0.7),
            ),
          ),
        ],
      ),
    );
  }
}

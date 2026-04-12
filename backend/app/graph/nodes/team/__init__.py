"""
Team subgraph node modules.

Splits the monolithic team_speak.py into 4 discrete subgraph nodes:
  - search_node: RAG/legal API evidence search per agent
  - discuss_node: internal team discussion with voting
  - statement_node: representative statement generation
  - assign_roles is imported directly from nodes/assign_roles.py
"""

from app.graph.nodes.team.search_node import search_node
from app.graph.nodes.team.discuss_node import discuss_node
from app.graph.nodes.team.statement_node import statement_node

__all__ = ["search_node", "discuss_node", "statement_node"]

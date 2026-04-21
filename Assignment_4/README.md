# AI Assignment Scheduling: Greedy vs A* Search

This repository contains a Python implementation of a dependency-constrained scheduling algorithm. The project explores and compares different Artificial Intelligence search and heuristic strategies to schedule a set of tasks (assignments) for a group of students, subject to prerequisite constraints and varying food costs.

## Problem Statement
A set of assignments must be completed by a group of students. Each assignment:
- Requires specific inputs (books/notes).
- Produces a specific output that might be a prerequisite for another assignment.
- Has a specific difficulty level mapped to a required "food item" cost.

The goal is to find a valid daily schedule that respects a maximum group size ($g$) per day and prerequisite dependencies, while minimizing the total number of days taken to complete all tasks.

## Implemented Algorithms

### 1. Greedy Approaches
Four distinct greedy heuristics are implemented to select the next available assignments:
* **Dependency Depth (Critical Path):** Prioritizes assignments that unlock the longest downstream path of prerequisites.
* **Earliest Deadline (Topological):** Prioritizes assignments closest to the initial inputs (shallowest level).
* **Food Cost:** Prioritizes assignments with the cheapest required food item.
* **Food Type Frequency:** Prioritizes assignments whose required food item appears most frequently among the remaining tasks.

### 2. A* Search
An $A^*$ search algorithm guarantees the mathematically optimal schedule (minimum total days). 
* **State:** Defined by the current day, completed assignments, and available outputs.
* **Heuristic $h(n)$:** `(remaining_assignments * group_size)`. This admissible heuristic assumes perfect parallelization to guarantee the shortest possible schedule without overestimating.

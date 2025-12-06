import json

data = json.load(open('localization_metrics_summary.json'))
results = data['all_results']

both_correct = sum(1 for r in results if r['file_correct'] and r['node_line_recall'] > 0)
total = len(results)

print(f'Total tasks: {total}')
print(f'Both file AND node-level correct: {both_correct} ({both_correct/total:.1%})')
print(f'\nBreakdown:')
print(f'  File correct only: {sum(1 for r in results if r["file_correct"] and r["node_line_recall"] == 0)}')
print(f'  Node correct only: {sum(1 for r in results if not r["file_correct"] and r["node_line_recall"] > 0)}')
print(f'  Both correct: {both_correct}')
print(f'  Neither: {sum(1 for r in results if not r["file_correct"] and r["node_line_recall"] == 0)}')


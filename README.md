# Skynet_DS

An automated, domain-specific GitHub App Webhook Service designed to serve as an automated infrastructure review gatekeeper during the Pull Request phase. Instead of auditing basic code style, this tool targets high-value, resource-intensive engineering flaws in Python data science and machine learning pipelines (Pandas, NumPy, PyTorch).

By combining lightweight Abstract Syntax Tree (AST) parsing via Tree-sitter with a targeted LLM Agent, the tool catches massive performance traps—such as vectorization failures (.iterrows()), hidden memory duplication, unbatched training inputs, and GPU context/leak omissions (with torch.no_grad()). It then directly injects precise, cost-saving code rewrites as native GitHub inline suggestions.

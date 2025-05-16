tactic_map = {
    "TA0043": "Reconnaissance",
    "TA0042": "Resource Development",
    "TA0001": "Initial Access",
    "TA0002": "Execution",
    "TA0003": "Persistence",
    "TA0004": "Privilege Escalation",
    "TA0005": "Defense Evasion",
    "TA0006": "Credential Access",
    "TA0007": "Discovery",
    "TA0008": "Lateral Movement",
    "TA0009": "Collection",
    "TA0011": "Command and Control",
    "TA0010": "Exfiltration",
    "TA0040": "Impact"
}

casual_model_graph = {
    "TA0043": ["TA0042"],
    "TA0042": ["TA0001"],
    "TA0001": ["TA0002", "TA0003"],
    "TA0002": ["TA0003", "TA0004", "TA0005", "TA0006", "TA0007", "TA0008", "TA0009", "TA0011"],
    "TA0003": ["TA0004", "TA0005", "TA0006", "TA0007", "TA0008", "TA0009", "TA0011"],
    "TA0004": ["TA0005", "TA0006", "TA0007", "TA0008", "TA0009", "TA0011"],
    "TA0005": ["TA0006", "TA0007", "TA0008", "TA0009", "TA0011"],
    "TA0006": ["TA0007", "TA0008", "TA0009", "TA0011"],
    "TA0007": ["TA0008", "TA0009", "TA0011"],
    "TA0008": ["TA0009", "TA0011"],
    "TA0009": ["TA0011"],
    "TA0011": ["TA0010", "TA0040","TA0002"],
    "TA0010": ["TA0040"],
    "TA0040": ["TA0043"]
}
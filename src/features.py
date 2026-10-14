# src/features.py

def clean_text(text):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return text.strip().lower()

def build_advanced_features(samples):
    ids = []
    contexts = []
    
    for s in samples:
        ids.append(s.get("id", ""))
        
        curr = clean_text(s.get("current_prompt", ""))
        
        history_list = s.get("history", [])[-3:] 
        history_tokens = []
        for h in history_list:
            act = clean_text(h.get("action", "none"))
            res = clean_text(h.get("result", "none"))
            history_tokens.append(f"past_action:{act} past_result:{res[:50]}")
        history_str = " ".join(history_tokens)
        
        meta = s.get("session_meta", {})
        meta_tokens = []
        for k, v in meta.items():
            meta_tokens.append(f"meta_{k}:{clean_text(str(v))}")
        meta_str = " ".join(meta_tokens)
        
        full_context = f"{curr} [HISTORY] {history_str} [META] {meta_str}"
        contexts.append(full_context)
        
    return ids, contexts
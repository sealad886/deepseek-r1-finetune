import os
from datasets import load_dataset
from config import config

def preprocess_dataset():
    ds = load_dataset(config.DEFAULT_DATASET, config.DEFAULT_NAME, trust_remote_code=True)

    def add_system_prompt(messages: list[dict[str: str]]):
        has_system = any(msg.get('role') == 'system' for msg in messages)
        if not has_system:
            system_msg = {'role': 'system', 'content': config.system_prompt}
            messages.insert(0, system_msg)
        else:
            existing_system = next(msg for msg in messages if msg.get('role') == 'system')
            existing_system['content'] = config.system_prompt + "\n" + existing_system['content']
        return messages

    def add_reasoning(messages, reasoning = None, answer = None):
        msg_content = ''
        if reasoning:
            msg_content += "<think>" + reasoning + "\n</think>"
        if answer:
            if reasoning: msg_content += "\n"
            msg_content += answer
        messages.append({
            'role': 'assistant',
            'content': msg_content
        })
        return messages

    def map_messages(columns_dict):
        messages = columns_dict.get("messages", [{},])
        reasoning = columns_dict.get("reasoning", "")
        answer = columns_dict.get("answer", "")
        return {
            "messages": add_reasoning(add_system_prompt(messages), reasoning, answer)
        }

    mapped_ds = ds.map(
        map_messages,
        # remove_columns=['model'],
        desc="Preprocess (to messages)"
        )
    for split in mapped_ds.keys():
        mapped_ds[split] = mapped_ds[split].rename_column("model", "labels").class_encode_column("labels")

    return mapped_ds

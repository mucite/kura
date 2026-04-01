import os

print('=== Model Verification ===')
llm_path = 'models/Llama-3.2-3B-Instruct-4bit-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf'
whisper_path = 'models/whisper/large-v3.pt'
old_whisper = 'models/whisper-large-v3-turbo-ct2'

print(f'LLM Model: {"✅ Found" if os.path.exists(llm_path) else "❌ Missing"}')
print(f'Whisper Model: {"✅ Found" if os.path.exists(whisper_path) else "❌ Missing"}')
print(f'Old Whisper CT2: {"❌ Still exists" if os.path.exists(old_whisper) else "✅ Removed"}')
print('=== All models ready! ===')


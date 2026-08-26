from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="",
)

# First API call with reasoning
response = client.chat.completions.create(
  model="dots-studio/dots-3-note-preview:free",
  messages=[
          {
            "role": "user",
            "content": "What is Genshin impact?"
          }
        ],
  extra_body={"reasoning": {"enabled": True}}
)

# Extract the assistant message with reasoning_details
response = response.choices[0].message
# print(response.choices[0].message.content)
print(response.reasoning_details)  # This will contain the reasoning details from the model
# print(getattr(response, "reasoning_details", None))  # This will also contain the reasoning details from the model
print(getattr(response, "content", None))  # This will contain the content of the assistant's message

# # Preserve the assistant message with reasoning_details
# messages = [
#   {"role": "user", "content": "What is Genshin impact?"},
#   {
#     "role": "assistant",
#     "content": response.content,
#     "reasoning_details": response.reasoning_details  # Pass back unmodified
#   },
#   {"role": "user", "content": "Are you sure? Think carefully."}
# ]

# # Second API call - model continues reasoning from where it left off
# response2 = client.chat.completions.create(
#   model="dots-studio/dots-3-note-preview:free",
#   messages=messages,
#   extra_body={"reasoning": {"enabled": True}}
# )
# print(response2.choices[0].message.content)
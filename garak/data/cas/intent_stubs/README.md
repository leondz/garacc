# Intent Stubs

* **Naming** Filenames must start with the intent code or intent prefix they cover, e.g. `S001` or `C006temporal`
* **Content** Files should contain one intent per line. Intents must be lowercase and start with an imperative verb, e.g. *write output*

# Formats

Intent stub sets can be represented multiple ways:

* .txt file -- One stub per line. Multiline not supported.
* .yaml file -- List of list of dicts, matching OpenAI API format and `garak.attempt.Conversation.from_dict()`
* .json file -- Two formats:
  * List of conversations: As per yaml file, a list of entries matching OpenAI / `Conversation.from_dict()` expectations
  * List of strings: One stub per entry - useful for simple, multiline stub datasets
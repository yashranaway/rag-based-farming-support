import replicate

input = {
    "prompt": "How is perplexity measured for LLMs and why is it useful?"
}

for event in replicate.stream(
    "ibm-granite/granite-3.3-8b-instruct",
    input=input
):
    print(event, end="")

from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """# Needle In A Haystack - Pressure Testing LLMs

Supported model providers: OpenAI, Anthropic

A simple 'needle in a haystack' analysis to test in-context retrieval ability of long context LLMs.

Get the behind the scenes on the [overview video](https://youtu.be/KwRRuiCCdmc).

![GPT-4-128 Context Testing](img/NeedleHaystackCodeSnippet.png)

git clone https://github.com/gkamradt/LLMTest_NeedleInAHaystack.git

## The Test
1. Place a random fact or statement (the 'needle') in the middle of a long context window (the 'haystack')
2. Ask the model to retrieve this statement
3. Iterate over various document depths (where the needle is placed) and context lengths to measure performance
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=300,
    chunk_overlap=0
)

chunk = splitter.split_text(text)

print(len(chunk))

print(chunk[0])
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """This is a long text that needs to be split into smaller chunks.

The RecursiveCharacterTextSplitter will split the text based on the specified chunk size and overlap.
It will try to split the text at natural break points, such as sentences or paragraphs, to ensure that the chunks are coherent and meaningful.

The chunk size can be adjusted to control how large each chunk is, and the overlap can be set to ensure that there is some context shared between the chunks.
This is particularly useful for tasks like language modeling or information retrieval, where having coherent chunks of text can improve the performance of the model."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap= 0,
)

chunks= splitter.split_text(text)

print(len(chunks))
print(chunks)
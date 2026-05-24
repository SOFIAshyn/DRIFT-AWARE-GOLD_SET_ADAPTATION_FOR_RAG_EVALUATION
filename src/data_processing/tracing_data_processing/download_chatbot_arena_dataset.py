from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("lmsys/chatbot_arena_conversations")

    print(dataset["train"][:5])

    dataset["train"].to_csv(
        "/Users/s.petryshyn/Desktop/UNI/COURSE WORK/data/raw/chatbot_arena.csv"
    )
    print("Dataset saved to /Users/s.petryshyn/Desktop/UNI/COURSE WORK/data/raw/chatbot_arena.csv")
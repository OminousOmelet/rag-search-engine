import argparse

from lib.multimodal_search import verify_image_embedding, image_search_command

def main():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser("verify_image_embedding", help="verify image embedding")
    verify_parser.add_argument("image_path", type=str, help="path to image file")
    img_search_parser = subparsers.add_parser("image_search", help="Search movies based on image")
    img_search_parser.add_argument("image_path", type=str, help="path to image file")
    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            documents = image_search_command(args.image_path)
            for i, doc in enumerate(documents, 1):
                print(f"{i}. {doc['title']} (similarity: {str(doc['cosine_sim'])[:5]})")
                print(f"{doc['description'][:100]}...\n")
        case _:
            parser.print_help()
            

if __name__ == "__main__":
    main()
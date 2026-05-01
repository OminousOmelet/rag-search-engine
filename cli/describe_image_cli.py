import argparse

from lib.describe_image import rewrite_query_for_img

def main():
    parser = argparse.ArgumentParser(description="Image description CLI")
    parser.add_argument("--image", type=str, help="path to image file")
    parser.add_argument("--query", type=str, help="query related to image")

    args = parser.parse_args()
    rewrite_query_for_img(args.image, args.query)

if __name__ == "__main__":
    main()
import os


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    web_dir = os.path.join(base_dir, "web")
    partials_dir = os.path.join(web_dir, "partials")

    # Order of concatenation
    parts = [
        "header.html",
        "home.html",
        "rf_explorer.html",
        "slice_explorer.html",
        "footer.html",
    ]

    output_content = ""

    print("Building index.html from partials...")
    for part in parts:
        path = os.path.join(partials_dir, part)
        if os.path.exists(path):
            print(f"  Adding {part}")
            with open(path, "r") as f:
                output_content += f.read() + "\n"
        else:
            print(f"  Warning: {part} not found!")

    output_path = os.path.join(base_dir, "index.html")
    with open(output_path, "w") as f:
        f.write(output_content)

    print(f"Successfully generated {output_path}")


if __name__ == "__main__":
    main()

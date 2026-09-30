"""License notices carried by generated support functions."""

SOURCE_LICENSES = {
    "fdlibm": (
        "Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.\n"
        "Developed at SunPro, a Sun Microsystems, Inc. business.\n"
        "Permission to use, copy, modify, and distribute this software is freely\n"
        "granted, provided that this notice is preserved."
    ),
}


def source_license_comments(ast, target="directx"):
    """Render each explicitly required notice once in the target comment syntax."""
    notices = set()
    if hasattr(ast, "walk"):
        for node in ast.walk():
            notices.update(node.annotations.get("source_licenses", ()))
    prefix = {"mojo": "#", "vulkan": ";"}.get(target, "//")
    return "".join(
        "".join(f"{prefix} {line}\n" for line in SOURCE_LICENSES[name].splitlines())
        for name in sorted(notices)
    )

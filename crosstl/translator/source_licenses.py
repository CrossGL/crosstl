"""License notices carried by generated support functions."""

SOURCE_LICENSES = {
    "softfloat": (
        "Berkeley SoftFloat Release 3e by John R. Hauser.\n"
        "Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018 The Regents of the\n"
        "University of California.  All rights reserved.\n"
        "Redistribution and use in source and binary forms, with or without\n"
        "modification, are permitted provided that the following conditions are met:\n"
        "1. Redistributions of source code must retain the above copyright notice,\n"
        "   this list of conditions, and the following disclaimer.\n"
        "2. Redistributions in binary form must reproduce the above copyright\n"
        "   notice, this list of conditions and the following disclaimer in the\n"
        "   documentation and/or other materials provided with the distribution.\n"
        "3. Neither the name of the University nor the names of its contributors\n"
        "   may be used to endorse or promote products derived from this software\n"
        "   without specific prior written permission.\n"
        'THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY\n'
        "EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED\n"
        "WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE\n"
        "DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY\n"
        "DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES\n"
        "(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;\n"
        "LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND\n"
        "ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT\n"
        "(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF\n"
        "THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
    ),
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

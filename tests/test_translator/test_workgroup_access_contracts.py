import pytest

from crosstl.project import (
    load_project_config,
    translate_project,
    validate_project_report,
)
from crosstl.translator.codegen.workgroup_access_contracts import (
    WorkgroupAccessAssertion,
    parse_workgroup_access_assertions,
)


def test_workgroup_access_assertion_parses_config_and_report_shapes():
    config_assertion = parse_workgroup_access_assertions(
        [
            {
                "source": "kernels/*.metal",
                "entry_point": "fft_mem_256_*",
                "function": "ReadWriter_*",
                "parameter": "buffer",
                "minimum": 0,
                "maximum": 255,
            }
        ]
    )[0]

    assert config_assertion == WorkgroupAccessAssertion(
        source="kernels/*.metal",
        entry_point="fft_mem_256_*",
        function="ReadWriter_*",
        parameter="buffer",
        minimum=0,
        maximum=255,
    )
    assert config_assertion.applies_to(
        "fft_mem_256_float2",
        "ReadWriter_float2",
        "buffer",
    )
    assert not config_assertion.applies_to(
        "fft_mem_512_float2",
        "ReadWriter_float2",
        "buffer",
    )
    assert parse_workgroup_access_assertions([config_assertion.to_json()]) == (
        config_assertion,
    )


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            {"minimum": 0, "maximum": 7},
            "entry_point is required",
        ),
        (
            {"entry_point": "main", "minimum": 8, "maximum": 7},
            "minimum must not exceed maximum",
        ),
        (
            {
                "entry_point": "main",
                "entryPoint": "main",
                "minimum": 0,
                "maximum": 7,
            },
            "must not define both entry_point and entryPoint",
        ),
    ],
)
def test_workgroup_access_assertion_rejects_invalid_records(record, message):
    with pytest.raises(ValueError, match=message):
        parse_workgroup_access_assertions([record])


def test_workgroup_access_assertion_rejects_non_integer_direct_bounds():
    with pytest.raises(ValueError, match="bounds must be integers"):
        WorkgroupAccessAssertion(
            entry_point="main",
            minimum=0.0,
            maximum=7,
        )


def test_project_workgroup_access_assertion_is_reported_and_consumed(tmp_path):
    source = """shader ProjectWorkgroupAccess {
    void store_value(threadgroup float* values, uint index) {
        values[index] = 1.0;
    }

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        void main(uint3 group @ gl_WorkGroupID) {
            threadgroup float sharedValues[8];
            store_value(sharedValues, group.x);
        }
    }
}
"""
    (tmp_path / "kernel.cgl").write_text(source, encoding="utf-8")
    (tmp_path / "crosstl.toml").write_text(
        """[project]
include = ["kernel.cgl"]
targets = ["opengl"]
output_dir = "out"

[[project.workgroup_access_assertions]]
source = "kernel.cgl"
entry_point = "main"
function = "store_value"
parameter = "values"
minimum = 0
maximum = 7
""",
        encoding="utf-8",
    )

    config = load_project_config(tmp_path)
    report = translate_project(config, format_output=False)
    payload = report.to_json()
    expected = {
        "source": "kernel.cgl",
        "entryPoint": "main",
        "function": "store_value",
        "parameter": "values",
        "minimum": 0,
        "maximum": 7,
    }

    assert [
        assertion.to_json() for assertion in config.workgroup_access_assertions
    ] == [expected]
    assert payload["project"]["workgroupAccessAssertions"] == [expected]
    assert payload["project"]["workgroupAccessAssertionCount"] == 1
    assert payload["summary"]["translatedCount"] == 1

    report_path = tmp_path / "report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True

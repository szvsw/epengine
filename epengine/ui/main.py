"""Main UI for the component library builder."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
from epinterface.sbem.common import MetadataMixin, NamedObject
from epinterface.sbem.components.composer import (
    construct_composer_model,
    construct_graph,
)
from epinterface.sbem.components.materials import EnvironmentalMixin
from epinterface.sbem.components.zones import ZoneComponent
from epinterface.sbem.prisma.client import Prisma, PrismaSettings, deep_fetcher
from streamlit.elements.lib.column_types import ColumnConfig

st.set_page_config(
    page_title="SBEM Component Library Builder",
    layout="wide",
)
st.title("SBEM Component Library Builder")


@st.cache_resource
def load_db(db_file, file_path: Path):
    """Load the database file."""
    with open(file_path, "wb") as f:
        f.write(db_file.getvalue())
    return PrismaSettings(
        database_path=file_path,
        auto_register=False,
    )


@st.cache_resource
def load_graph():
    """Load the graph."""
    g = construct_graph(
        root_node=ZoneComponent,
    )
    SelectorModel = construct_composer_model(
        g,
        root_validator=ZoneComponent,
        use_children=False,
    )
    return g, SelectorModel


@st.cache_resource
def load_schema():
    """Load the schema."""
    return ZoneComponent.model_json_schema()


db_file = st.file_uploader("Upload a database file.", type="db")

st.divider()


def handle_prop_type(comp_name, prop_name, prop_type, prop_info, with_null: bool):
    """Handle the property type."""
    help_text = prop_info.get("description", "")
    if with_null:
        st.error(f"Null is not supported: {comp_name}:`{prop_name}`")
        return
    if "enum" in prop_info:
        st.selectbox(
            prop_name,
            prop_info["enum"],
            key=f"{comp_name}_{prop_name}_selectbox",
            help=help_text,
        )
    elif prop_type == "string":
        st.text_input(prop_name, key=f"{comp_name}_{prop_name}_string", help=help_text)
    elif prop_type == "integer":
        st.number_input(
            prop_name,
            min_value=prop_info.get("minimum", None),
            max_value=prop_info.get("maximum", None),
            step=1,
            help=help_text,
        )
    elif prop_type == "number":
        minimum = prop_info.get("minimum", None)
        maximum = prop_info.get("maximum", None)
        st.number_input(prop_name, min_value=minimum, max_value=maximum, help=help_text)
    elif prop_type == "boolean":
        st.toggle(prop_name, help=help_text)
    else:
        st.write(f"Unhandled prop type: `{prop_name}`:`({prop_type})`")


def handle_prop(comp_name, prop_name, prop_info, with_null: bool = False):
    """Handle the property."""
    if "anyOf" in prop_info:
        if len(prop_info["anyOf"]) == 2 and prop_info["anyOf"][-1] == {"type": "null"}:
            handle_prop(comp_name, prop_name, prop_info["anyOf"][0], with_null=True)
        else:
            for case_def in prop_info["anyOf"]:
                handle_prop(comp_name, prop_name, case_def, with_null=with_null)
    elif "$ref" in prop_info:
        ref: str = prop_info["$ref"]
        help_text = prop_info.get("description", "")
        ref_name = ref.split("/")[-1]
        opts: list[str | None] = ["Option 1", "Option 2"]
        if with_null:
            opts.append(None)
        st.selectbox(
            f"{prop_name} `({ref_name})`:",
            opts,
            key=f"{comp_name}_{prop_name}_selectbox",
            help=help_text,
        )
    elif "type" in prop_info:
        handle_prop_type(comp_name, prop_name, prop_info["type"], prop_info, with_null)
    else:
        st.write(prop_name)


def handle_component(component: str, defn: dict):
    """Handle the component."""
    component_name = defn.get("title", component)
    with st.expander(component_name, expanded=True):
        with st.form(key=f"create_{component}"):
            columns = st.columns(3)
            i = -1
            for prop, info in defn["properties"].items():
                if prop not in [
                    *EnvironmentalMixin.model_fields,
                    *MetadataMixin.model_fields,
                ]:
                    i += 1
                    prop_name = info.get("title", prop)
                    with columns[i % 3]:
                        handle_prop(component, prop_name, info)
            _submitted = st.form_submit_button(
                "Save", use_container_width=True, type="primary"
            )
        st.subheader("def")
        st.write(defn)


def handle_schema(schema: dict):
    """Handle the schema."""
    for component, defn in schema["$defs"].items():
        handle_component(component, defn)


@dataclass
class HelpTableConfig:
    """Help table config."""

    title: str
    help_text: str
    dtype: str
    column_info: dict
    required: bool
    is_ref: bool


def handle_data_table_column(
    col_name: str,
    column_info: dict,
    classes: dict[str, type[NamedObject]],
    nullable: bool = False,
) -> tuple[ColumnConfig, HelpTableConfig]:
    """Handle the data table column."""
    col_title = column_info.get("title", col_name)
    help_text = column_info.get("description", "")
    editable = col_name not in ["Name"]
    editable = True
    _is_ref = False
    if "$ref" in column_info:
        _is_ref = True
        ref: str = column_info["$ref"]
        ref_name = ref.split("/")[-1]
        fetcher = deep_fetcher.get_deep_fetcher(classes[ref_name])
        opts = fetcher.prisma_model.prisma(db).find_many(include=fetcher.include)
        opts = [o.Name for o in opts]
        help_table_config = HelpTableConfig(
            title=col_title,
            help_text=help_text,
            dtype=ref_name,
            column_info=column_info,
            required=not nullable,
            is_ref=True,
        )
        return st.column_config.SelectboxColumn(
            col_title,
            help=help_text,
            options=opts,
            required=not nullable,
            disabled=not editable,
        ), help_table_config
    elif (
        "anyOf" in column_info
        and len(column_info["anyOf"]) == 2
        and column_info["anyOf"][-1] == {"type": "null"}
    ):
        return handle_data_table_column(
            col_name, column_info["anyOf"][0], classes, nullable=True
        )
    elif "enum" in column_info:
        help_table_config = HelpTableConfig(
            title=col_title,
            help_text=help_text,
            dtype="|".join(column_info["enum"]),
            column_info=column_info,
            required=not nullable,
            is_ref=False,
        )
        return st.column_config.SelectboxColumn(
            col_title,
            help=help_text,
            options=column_info["enum"],
            required=not nullable,
            disabled=not editable,
        ), help_table_config
    elif "type" in column_info and column_info["type"] == "boolean":
        help_table_config = HelpTableConfig(
            title=col_title,
            help_text=help_text,
            dtype="bool",
            column_info=column_info,
            required=not nullable,
            is_ref=False,
        )
        return st.column_config.CheckboxColumn(
            col_title,
            help=help_text,
            required=not nullable,
            disabled=not editable,
        ), help_table_config
    elif "type" in column_info and column_info["type"] == "integer":
        minimum = column_info.get("minimum")
        maximum = column_info.get("maximum")
        bounds_str = (
            f":{minimum or '-inf'} <= x <= {maximum or 'inf'}"
            if minimum is not None and maximum is not None
            else ""
        )
        help_table_config = HelpTableConfig(
            title=col_title,
            help_text=help_text,
            dtype="int" + bounds_str,
            column_info=column_info,
            required=not nullable,
            is_ref=False,
        )
        return st.column_config.NumberColumn(
            col_title,
            help=help_text,
            min_value=minimum,
            max_value=maximum,
            step=1,
            required=not nullable,
            disabled=not editable,
        ), help_table_config
    elif "type" in column_info and column_info["type"] == "number":
        minimum = column_info.get("minimum")
        maximum = column_info.get("maximum")
        bounds_str = (
            f":{minimum or '-inf'} <= x <= {maximum or 'inf'}"
            if minimum is not None and maximum is not None
            else ""
        )
        help_table_config = HelpTableConfig(
            title=col_title,
            help_text=help_text,
            dtype="float" + bounds_str,
            column_info=column_info,
            required=not nullable,
            is_ref=False,
        )
        return st.column_config.NumberColumn(
            col_title,
            help=help_text,
            min_value=minimum,
            max_value=maximum,
            required=not nullable,
            disabled=not editable,
        ), help_table_config
    elif "type" in column_info and column_info["type"] == "string":
        help_table_config = HelpTableConfig(
            title=col_title,
            help_text=help_text,
            dtype="str",
            column_info=column_info,
            required=not nullable,
            is_ref=False,
        )

        return st.column_config.TextColumn(
            col_title,
            help=help_text,
            required=not nullable,
            disabled=not editable,
        ), help_table_config
    elif "type" in column_info and column_info["type"] == "array":
        st.warning("Array type not supported.")
        help_table_config = HelpTableConfig(
            title=col_title,
            help_text=help_text,
            dtype="array",
            column_info=column_info,
            required=not nullable,
            is_ref=False,
        )
        return st.column_config.ListColumn(
            col_title,
            help=help_text,
            # required=True,
            # disabled=not editable,
        ), help_table_config
    else:
        st.warning(f"{col_name} has an unsupported data entry type.")
        help_table_config = HelpTableConfig(
            title=col_title,
            help_text=help_text,
            dtype="unsupported",
            column_info=column_info,
            required=not nullable,
            is_ref=False,
        )
        return st.column_config.TextColumn(
            col_title,
            help=help_text,
            required=not nullable,
            disabled=not editable,
        ), help_table_config


def handle_data_table(
    db: Prisma,
    schema: dict,
    component_class: type[NamedObject],
    classes: dict[str, type[NamedObject]],
):
    """Handle the data table."""
    fetcher = deep_fetcher.get_deep_fetcher(component_class)
    defn: dict[str, Any] = schema["$defs"][component_class.__name__]
    component_class_title = defn.get("title", component_class.__name__)
    component_class_description = defn.get("description", "")

    records = fetcher.prisma_model.prisma(db).find_many(include=fetcher.include)
    original_record_names = [record.Name for record in records]
    records = [
        {
            k: v if not isinstance(v, dict) else v.get("Name")
            for k, v in record.model_dump().items()
            if k in defn["properties"]
        }
        for record in records
    ]
    column_configs_with_ref_flag = {
        k: handle_data_table_column(k, v, classes)
        for k, v in defn["properties"].items()
    }
    column_configs = {k: v[0] for k, v in column_configs_with_ref_flag.items()}
    help_table_configs = {k: v[1] for k, v in column_configs_with_ref_flag.items()}
    column_config = {
        **dict.fromkeys(MetadataMixin.model_fields),
        **dict.fromkeys(EnvironmentalMixin.model_fields),
        **column_configs,
    }
    with st.form(key=f"edit_{selected_component}_dt"):
        st.header(component_class_title)
        st.text(component_class_description)
        dt = st.data_editor(
            records,
            # column_config={
            #     **dict.fromkeys(MetadataMixin.model_fields),
            #     **dict.fromkeys(EnvironmentalMixin.model_fields),
            # },
            column_config=column_config,
            num_rows="dynamic",
        )
        dt_with_connectors = [
            {
                **record,
                **{
                    k: {"connect": {"Name": v}}
                    for k, v in record.items()
                    if help_table_configs[k].is_ref
                },
            }
            for record in dt
        ]
        submitted = st.form_submit_button(
            "Save", use_container_width=True, type="primary"
        )
        if submitted:
            with db.tx() as tx:
                prisma_model = fetcher.prisma_model

                for i, record in enumerate(dt_with_connectors):
                    if i < len(records):
                        if record["Name"] != original_record_names[i]:
                            msg = f"Names of records which have already been created cannot be changed: {record['Name']} != {original_record_names[i]}"
                            st.error(msg)
                            raise ValueError(msg)
                        prisma_model.prisma(tx).update(
                            where={"Name": record["Name"]},
                            data=record,
                            include=fetcher.include,
                        )

                    else:
                        prisma_model.prisma(tx).create(
                            data=record,
                            include=fetcher.include,
                        )

                    fetcher.get_deep_object(record["Name"], tx)
    with st.expander("Schema", expanded=True):
        help_data = {
            k: {
                "Description": v.help_text,
                "Type": f"`{v.dtype}`",
                "Required": f"{v.required}",
            }
            for k, v in help_table_configs.items()
        }
        st.write(help_table_configs)
        st.table(help_data)


if db_file is not None:
    schema = load_schema()
    g, SelectorModel = load_graph()
    classes: dict[str, type[NamedObject]] = {}

    # for node in g.nodes:
    #     direct_children = g.successors(node)
    #     st.divider()
    #     st.text(node)
    # for child in direct_children:
    #     print(child)
    for edge in g.edges(data=True):
        source_node, target_node, data = edge
        cls = data["data"]["type"]
        cls_name = cls.__name__
        if cls_name not in classes:
            classes[cls_name] = cls

    selected_component = st.selectbox("Component type:", list(classes.keys()))

    selected_component_class = classes[selected_component]
    with load_db(db_file, Path("db.db")).db as db:
        handle_data_table(db, schema, selected_component_class, classes)


st.divider()


# def old_editor():
#     """Old editor."""
#     if db_file is not None:
#         with load_db(db_file, Path("db.db")).db as db:
#             g, SelectorModel = load_graph()
#             classes = {}

#             # for node in g.nodes:
#             #     direct_children = g.successors(node)
#             #     st.divider()
#             #     st.text(node)
#             # for child in direct_children:
#             #     print(child)
#             for edge in g.edges(data=True):
#                 source_node, target_node, data = edge
#                 cls = data["data"]["type"]
#                 cls_name = cls.__name__
#                 if cls_name not in classes:
#                     classes[cls_name] = cls

#             component_type_col, component_mode_col, component_name_col = st.columns([
#                 1,
#                 1,
#                 3,
#             ])
#             with component_type_col:
#                 selected_component = st.selectbox(
#                     "Component type:", list(classes.keys())
#                 )

#             selected_component_class = classes[selected_component]

#             with component_mode_col:
#                 selected_component_mode = st.selectbox(
#                     "Component mode:", ["Edit", "Create"]
#                 )

#             if selected_component_mode == "Edit":
#                 try:
#                     fetcher = deep_fetcher.get_deep_fetcher(selected_component_class)
#                     pm = fetcher.prisma_model
#                     include = fetcher.include
#                     records = pm.prisma(db).find_many(include=include)
#                     with component_name_col:
#                         selected_component_name = st.selectbox(
#                             f"Edit an existing {selected_component}.",
#                             [record.Name for record in records],
#                             key=f"edit_existing_{selected_component}",
#                         )
#                 except ValueError as e:
#                     if "No link found for" in str(e):
#                         pass
#                     else:
#                         raise

#                 else:
#                     if len(records) > 0:
#                         record, comp = fetcher.get_deep_object(
#                             cast(str, selected_component_name), db
#                         )
#                         with st.form(
#                             key=f"edit_{selected_component}_{selected_component_name}"
#                         ):
#                             st.subheader(selected_component_name)
#                             # st.write(selected_component_name)
#                             n_fields = len(selected_component_class.model_fields) - 1
#                             if issubclass(selected_component_class, EnvironmentalMixin):
#                                 n_fields -= len(EnvironmentalMixin.model_fields)
#                             if issubclass(selected_component_class, MetadataMixin):
#                                 n_fields -= len(MetadataMixin.model_fields)
#                             form_cols = st.columns(min(3, n_fields), gap="large")
#                             col_ix = -1
#                             data = {}
#                             for (
#                                 field,
#                                 info,
#                             ) in selected_component_class.model_fields.items():
#                                 if (
#                                     field in MetadataMixin.model_fields
#                                     or field in EnvironmentalMixin.model_fields
#                                 ) or field == "Name":
#                                     continue
#                                 col_ix += 1
#                                 field_name = (
#                                     info.title if hasattr(info, "title") else field
#                                 )
#                                 help_text = (
#                                     info.description
#                                     if hasattr(info, "description")
#                                     else ""
#                                 )
#                                 with form_cols[col_ix % 3]:
#                                     # check if the type is a list
#                                     if (
#                                         hasattr(info.annotation, "__origin__")
#                                         and info.annotation.__origin__ is list  # pyright: ignore [reportOptionalMemberAccess]
#                                     ):
#                                         st.write(field_name)
#                                         st.write(info)
#                                         st.write("List type.")
#                                         st.write(help_text)
#                                     # check if the annotation is a class
#                                     elif isinstance(
#                                         info.annotation, type
#                                     ) and issubclass(info.annotation, NamedObject):
#                                         val = st.selectbox(
#                                             f"{field_name} `({info.annotation.__name__})`:",
#                                             [getattr(comp, field).Name],
#                                             help=help_text,
#                                         )
#                                         data[field] = {"connect": {"Name": val}}
#                                     elif info.annotation is bool:
#                                         st.write(field_name)
#                                         val = st.toggle(
#                                             cast(str, field_name),
#                                             value=getattr(comp, field),
#                                             help=help_text,
#                                         )
#                                         data[field] = val
#                                     elif info.annotation is str:
#                                         val = st.text_input(
#                                             cast(str, field_name),
#                                             value=getattr(comp, field),
#                                             help=help_text,
#                                         )
#                                         data[field] = val
#                                     elif info.annotation is float:
#                                         low, high = None, None
#                                         if info.metadata:
#                                             for meta in info.metadata:
#                                                 low_ = (
#                                                     meta.ge
#                                                     if hasattr(meta, "ge")
#                                                     else meta.geq
#                                                     if hasattr(meta, "geq")
#                                                     else None
#                                                 )
#                                                 if low_ is not None:
#                                                     low = float(low_)
#                                                 high_ = (
#                                                     meta.le
#                                                     if hasattr(meta, "le")
#                                                     else meta.leq
#                                                     if hasattr(meta, "leq")
#                                                     else None
#                                                 )
#                                                 if high_ is not None:
#                                                     high = float(high_)

#                                         val = st.number_input(  # pyright: ignore [reportCallIssue]
#                                             cast(str, field_name),
#                                             value=float(getattr(comp, field)),
#                                             min_value=low,
#                                             max_value=high,
#                                             help=help_text,
#                                         )
#                                         data[field] = val
#                                     # handle literal fields by checking if annotation is a Literal type
#                                     elif (
#                                         hasattr(info.annotation, "__origin__")
#                                         and info.annotation.__origin__ is Literal  # pyright: ignore [reportOptionalMemberAccess]
#                                     ):
#                                         val = st.selectbox(  # pyright: ignore [reportCallIssue]
#                                             cast(str, field_name),
#                                             options=info.annotation.__args__,  # pyright: ignore [reportOptionalMemberAccess]
#                                             index=info.annotation.__args__.index(  # pyright: ignore [reportOptionalMemberAccess]
#                                                 getattr(comp, field)
#                                             ),
#                                             help=help_text,
#                                         )
#                                         data[field] = val

#                                     else:
#                                         st.write(field_name)
#                                         st.write(info)
#                                         st.write("Unhandled field type.")
#                                         st.write(help_text)
#                             submitted = st.form_submit_button(
#                                 "Save",
#                                 use_container_width=True,
#                                 type="primary",
#                             )
#                             if submitted:
#                                 with db.tx() as tx:
#                                     pm.prisma(tx).update(
#                                         where={"Name": selected_component_name},
#                                         data=data,
#                                         include=include,
#                                     )
#                     else:
#                         st.warning("No existing component selected.")

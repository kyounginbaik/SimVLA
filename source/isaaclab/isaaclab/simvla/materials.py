"""Stage-local compatibility for sparse imported OmniPBR material definitions."""


def prepare_omnipbr_color_inputs(prims):
    """Author missing inputs used unconditionally by Replicator's color node.

    Imported Anubis shaders declare diffuse color but omit project_uvw. Isaac
    Sim 4.5's OgnSampleOmniPBR calls GetInput(...).Set without creating that
    input, raising UsdExpiredPrimAccessError. Edit only the current stage layer;
    do not save or alter the downloaded USD, bindings, textures or other shaders.
    """
    from pxr import Sdf, Usd, UsdShade

    updated = []
    seen = set()
    for prim in prims:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            continue
        for child in Usd.PrimRange(material.GetPrim()):
            if not child.IsA(UsdShade.Shader) or str(child.GetPath()) in seen:
                continue
            seen.add(str(child.GetPath()))
            if child.GetAttribute("info:mdl:sourceAsset:subIdentifier").Get() != "OmniPBR":
                continue
            shader = UsdShade.Shader(child)
            if not shader.GetInput("project_uvw"):
                shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(False)
                updated.append(str(child.GetPath()))
    return updated

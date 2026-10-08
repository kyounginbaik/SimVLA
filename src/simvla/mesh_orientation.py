"""SimVLA: per-mesh up axis — which way a BODex mesh actually stands.

Without an entry a mesh takes FALLBACK_UP, (0,1,0), unless its TYPE is listed in TYPE_UP. That is
right for the core_*/ddg_* meshes and WRONG for most sem_* ones, which store +Z up. The effect is
not subtle: 19 of 29 object types were being placed lying on their side, and a vase was resting on
0.1% of its surface -- it would topple the moment physics started.

Two callers resolve through resolved_up() below, and both must: kitchen_build._object_pose to place
the object, kitchen_gallery.mesh_tiles to draw the tile it is CHOSEN from.

PROVENANCE, because this table cannot be regenerated. Five automatic proxies have now been tried
and all five failed: bottom-contact area (a round-bottomed bowl touches at a point and is still
upright), trimesh stable poses (they lay bottles down and invert bowls, which is more stable but
not what a kitchen wants), a convex-hull side profile (it erases the opening that distinguishes a
vase from the same vase upside down), argmax bottom-contact over the six axes (measured against
the labels below: 67%, and its errors are systematic -- it inverts every open vessel, because a
vase rests on more of itself balanced on its rim than standing on its base, and it lays boxes on
their largest face), and BODex's own info/tabletop_pose.json (those are arbitrary resting poses
generated for grasp diversity, not canonical ones: 56 of 72 are not even axis-aligned). Uprightness
is semantic, not geometric. So every mesh here was labelled by hand from renders.

TWO PASSES, same method, different sheets:
  * 73 meshes from front/top/cross-section renders.
  * 53 more from six-candidate side renders -- the mesh stood on each of +/-X, +/-Y, +/-Z above a
    ground line, upright read off the sheet. The instrument was checked against pass one first: it
    reproduces the recorded vector for a vase, a plate, a candle and a teapot. This pass closed
    every remaining sem_* mesh that scene_spec.CATEGORIES offers, which is now a maintained
    invariant -- see test_mesh_orientation.test_no_offered_sem_mesh_is_left_to_the_fallback.
    52 of the 53 came out +Z; the exception is a toilet roll that stands on its end.

INDEPENDENTLY CONFIRMED afterwards, by a source nobody had looked at while the labelling was going
on: the pose BODex recorded when it synthesised each mesh's grasps (see GRASP FRAME below). It
agrees with 52 of those 53 -- the same toilet roll is the sole disagreement. It is a strong prior
for the axis and free to read, so check it FIRST if this table ever needs extending. It is not the
criterion, though: it puts a plate face-down at (0,0,1), where the human label says a plate rests
concave-up.

VERIFIED through the real scene_synthesizer.Asset pipeline, not by inspection: an entry either
raises the mesh's bottom-contact area (vase 0.1% -> 9.1%, candle 1.3% -> 15.0%) or stands the mesh
UP, which costs contact honestly -- a cereal box on its base touches far less of the counter than
the fallback did laying it on its largest face. Five meshes do neither and are pinned by name in
test_mesh_orientation.LOW_CONTACT_BY_DESIGN with the reason for each.

GRASP FRAME -- measured, and the answer is the opposite of what this note used to say. It warned
that plan_arm_grasp rotates grasps by the kitchen's Z angle alone (`rotated_data` from rot_z, no
`up` term), so re-orienting a mesh would move it out from under its own grasps, and concluded the
table was dangerous to read. Both halves of that are wrong, because the grasps are not authored in
the raw file frame.

Every BODex .npy carries the pose it was SYNTHESISED under, in world_cfg[0]["mesh"][...]["pose"]
(curobo wxyz). It is not identity: core_*/ddg_* were synthesised standing on +Y, sem_* on +Z. So a
grasp is already expressed in the frame of an object stood up that way, and rot_z alone is right --
PROVIDED the object is placed on the same axis it was synthesised on. Which is what this table
does. Over the 357 meshes that have grasps, placed up equals synthesised up for 340. Before the
table, every sem_* mesh was placed on +Y while its grasps assumed +Z: 90 degrees out. The table
CLOSES that gap rather than opening it, and no align(up -> +Z) correction belongs in the grasp
path -- adding one would break the 340 that already agree.

The 17 that DO disagree are handled in grasp_frame.py rather than here, and the labels below win.
Ten of them are hand-labelled (0,0,-1) against a (0,0,1) synthesis -- two plates, two food items, a
soap bar, two toilet rolls, a cookie, a pizza, a vase -- so their grasps arrive 180 degrees out
about X; three more are toilet rolls on a different axis again; four are sem_* meshes with no entry
at all, taking the core_* fallback (two Detergent, two ToiletPaper). Seven of the 17 are in
scene_spec.MESH_EXCLUDE and so can never reach a scene, which leaves 10 that can.

The labels win because the alternative is worse: BODex synthesised the plates face-DOWN, and
following its pose would place every plate convex side up. So grasp_frame rotates the grasps by
P @ S^-1 instead -- out of the synthesis frame, into the placed one -- which is exactly the identity
for the 340 that already agree. That correction is derived per mesh from the .npy, not from a list,
so changing a label here cannot leave a stale entry behind somewhere else.
"""

#: mesh directory name -> the vector in MESH coordinates that must point up in the world.
#: Anything absent keeps kitchen_build._object_pose's default. front must be chosen perpendicular
#: to up: (0,1,0) when up is +/-Z or +/-X, else (0,0,-1) -- the convention apple/sodacan already use.
MESH_UP: dict[str, tuple[int, int, int]] = {
    # Blender
    "sem_Blender_482ae495439fca7ef5afb395b99ae069": (0, 0, 1),
    "sem_Blender_48bddfc383e475583f36fda0f70067c5": (0, 0, 1),
    # Candle
    "sem_Candle_187d89b0c436d0edc8ab9067fbd9681d": (0, 0, 1),
    "sem_Candle_2109b61533489495c4cbe5e698cebc61": (0, 0, 1),
    "sem_Candle_2bdb2518ad90f679e86918abaa615eee": (0, 0, 1),
    "sem_Candle_3e6473aeac7d240bcbdb36da0ec8fda6": (0, 0, 1),
    "sem_Candle_88afa2933cf484378a8cf1e3e05903e3": (0, 0, 1),
    "sem_Candle_c29ceb4f2711534d9cbae40750f203a6": (0, 0, 1),
    "sem_Candle_da6658f36a9e81e8c4cbe5e698cebc61": (0, 0, 1),
    "sem_Candle_f13cd463079ccc3071fed78a5e58c3d1": (0, 0, 1),
    # Cap
    "sem_Cap_90c6bffdc81cedbeb80102c6e0a7618a": (0, 0, 1),
    "sem_Cap_c2cf2cf35d08662945c5fa74440a4519": (0, 0, 1),
    "sem_Cap_ed52d2c2a9c1b606cf9bbc203a5cff99": (0, 0, 1),
    # CerealBox
    "sem_CerealBox_12a565f8aecaafcbce41b639931f9ca1": (0, 0, 1),
    "sem_CerealBox_1bc88814b90df8c45a17ac5e1b4e4a4b": (0, 0, 1),
    "sem_CerealBox_1dd407598b5850959b1500745a428d00": (0, 0, 1),
    "sem_CerealBox_2ee85d45fe615a734322eb6f7ad3b3a2": (0, 0, 1),
    "sem_CerealBox_45628e5fceff8eda39f3410d5f76299b": (0, 0, 1),
    "sem_CerealBox_5062771877ee2a229eeafe87febc2d14": (0, 0, 1),
    "sem_CerealBox_a15f43d04b3d5256c9ea91c70932c737": (0, 0, 1),
    "sem_CerealBox_a61cd12446207107d59ff053d1480d84": (0, 0, 1),
    "sem_CerealBox_a69cca2394a1da2c3df318a0e4be8421": (0, 0, 1),
    "sem_CerealBox_a9beebe1b851adacd59ff053d1480d84": (0, 0, 1),
    "sem_CerealBox_abc86fa217ffeeeab3c769e22341ffb4": (0, 0, 1),
    "sem_CerealBox_b7d15c3a5784f95460c90262984aef06": (0, 0, 1),
    "sem_CerealBox_c87bb65a628e0ca6a4bd57b754f4233c": (0, 0, 1),
    "sem_CerealBox_dc394c7fdea7e07887e775fad2c0bf27": (0, 0, 1),
    "sem_CerealBox_defe74c05001775c13d6ba6128d0b598": (0, 0, 1),
    "sem_CerealBox_e481c3d08c11fc9e2978181091f6882a": (0, 0, 1),
    # Cookie
    "sem_Cookie_2ec24ca1dd42c39f6ad0370a38d43358": (0, 0, 1),
    "sem_Cookie_6a4bcdda0de2810190a3f8bc630957a8": (0, 0, -1),
    "sem_Cookie_a8e3118bec4c495f737a00f007529fbf": (0, 0, 1),
    "sem_Cookie_ccfa74e5574678325cde8c99e4b182f9": (0, 0, 1),
    # Cup
    "sem_Cup_dc1c220c8ef89a5d1a57566a9a7e5976": (0, 0, 1),
    # Detergent
    "sem_Detergent_547fa0085800c5c3846564a8a219239b": (0, 0, 1),
    "sem_Detergent_c02d623423a67ed9bda72093f9b5aa73": (0, 0, 1),
    "sem_Detergent_eded10bf44a2571a911cff0cb398f845": (0, 0, 1),
    # DrinkingUtensil
    "sem_DrinkingUtensil_128ecbc10df5b05d96eaf1340564a4de": (0, 0, 1),
    "sem_DrinkingUtensil_159e56c18906830278d8f8c02c47cde0": (0, 0, 1),
    "sem_DrinkingUtensil_181846dba8cfa3e91fc277ccdec7582": (0, 0, 1),
    "sem_DrinkingUtensil_187859d3c3a2fd23f54e1b6f41fdd78a": (0, 0, 1),
    "sem_DrinkingUtensil_1e227771ef66abdb4212ff51b27f0221": (0, 0, 1),
    "sem_DrinkingUtensil_1ea9ea99ac8ed233bf355ac8109b9988": (0, 0, 1),
    "sem_DrinkingUtensil_23fb2a2231263e261a9ac99425d3b306": (0, 0, 1),
    "sem_DrinkingUtensil_24651c3767aa5089e19f4cee87249aca": (0, 0, 1),
    "sem_DrinkingUtensil_2997f21fa426e18a6ab1a25d0e8f3590": (0, 0, 1),
    "sem_DrinkingUtensil_2e228ee528f0a7054212ff51b27f0221": (0, 0, 1),
    "sem_DrinkingUtensil_2e8d37b3dd65c312b8f0377fb16daf9d": (0, 0, 1),
    "sem_DrinkingUtensil_31ec660f1bdbec47ddcb31426795fd11": (0, 0, 1),
    # FoodItem
    "sem_FoodItem_118644ba80aa5048ac59fb466121cd17": (0, 0, 1),
    "sem_FoodItem_13b48456dad49f8762edbb9e1af21a03": (0, 0, 1),
    "sem_FoodItem_14d31ee208e1823f48b9fefd9341bfa": (0, 0, -1),
    "sem_FoodItem_16afaf11b6a9578444d3a157597f918b": (0, 0, 1),
    "sem_FoodItem_1a10ecfcaac04c882d17d82c03b66": (0, 0, 1),
    "sem_FoodItem_1fe6ed518ca583a77215a1e3ffbff428": (0, 0, 1),
    "sem_FoodItem_27f2de454f50ccea39f3410d5f76299b": (0, 0, 1),
    "sem_FoodItem_28b280b06401e17c83b787d4ec835b6b": (0, 0, -1),
    "sem_FoodItem_2cdbefbef5a1319aa72b7c690d778792": (0, 0, 1),
    "sem_FoodItem_2ddf6075c2e4fe46b5de9f9c43608e2e": (0, 0, 1),
    "sem_FoodItem_2e5a331b7d16f263d9e1e44903fea4a9": (0, 0, 1),
    "sem_FoodItem_32e7b785fc1bec024cf6350d5689a769": (0, 0, 1),
    # Fruit
    "sem_Fruit_b095a1c41447e9da887d2d6e4a6617b3": (0, 0, 1),
    # Kettle
    "sem_Kettle_9c5b0553b7a7088759ac5a8e79274552": (0, 0, 1),
    # MilkCarton
    "sem_MilkCarton_1c30b1102303f27ef690049a092c5efc": (0, 0, 1),
    "sem_MilkCarton_2546e080ae36044d25924baa64fe1af2": (0, 0, 1),
    "sem_MilkCarton_313026b3dd934a8c66ab797924e689f7": (0, 0, 1),
    "sem_MilkCarton_357348cf087fb03898feb9a919eb17de": (0, 0, 1),
    "sem_MilkCarton_3c426552b9df786b3f7f3ed2c3c690fe": (0, 0, 1),
    "sem_MilkCarton_49bdc6bed3f233eb3789e8d58849675c": (0, 0, 1),
    "sem_MilkCarton_64018b545e9088303dd0d6160c4dfd18": (0, 0, 1),
    "sem_MilkCarton_673d78d408bf142b9642f282fb76cb9": (0, 0, 1),
    "sem_MilkCarton_6ea87d3f4498448b3571657dafc7119": (0, 0, 1),
    "sem_MilkCarton_72110f88e2e51bf450e961cdc6fb5c30": (0, 0, 1),
    "sem_MilkCarton_e403646bb3e1e75850e961cdc6fb5c30": (0, 0, 1),
    "sem_MilkCarton_f5b5a24adc6826ace41b639931f9ca1": (0, 0, 1),
    "sem_MilkCarton_fa98055f6f2d0dd916b3af7b23ac8a74": (0, 0, 1),
    "sem_MilkCarton_fce427271b63011bfefca37858a6cef8": (0, 0, 1),
    # Pizza
    "sem_Pizza_99dadb83b9bcf2498af30108ea9ccb6c": (0, 0, 1),
    "sem_Pizza_b3a8ebcf6e9ca5852721058c5c629b05": (0, 0, -1),
    "sem_Pizza_df62f7b84fd210ba82cabcc4d06ff984": (0, 0, 1),
    # Plate
    "sem_Plate_1389e932a2c776d83c143af07c12991a": (0, 0, -1),
    "sem_Plate_9969f6178dcd67101c75d484f9069623": (0, 0, -1),
    # RiceCooker
    "sem_RiceCooker_1f00a773e33908a3abb564bb0657e0d6": (0, 0, 1),
    "sem_RiceCooker_5aa24fc0f25b61abb564bb0657e0d6": (0, 0, 1),
    "sem_RiceCooker_86175eb31a23a452757a72d9e80d1698": (0, 0, 1),
    "sem_RiceCooker_b3bbd60c89f54f1231a50841704a69bf": (0, 0, 1),
    # SoapBar
    "sem_SoapBar_351dec75ac0619b1827473663798726a": (0, 0, 1),
    "sem_SoapBar_5aba32dba149598911f4f6c1a9b8e604": (0, 0, -1),
    "sem_SoapBar_8da3c818adf836fcf2c4595240954986": (0, 0, 1),
    "sem_SoapBar_b3718bdc7025004813f5ec8466b8a488": (0, 0, 1),
    "sem_SoapBar_c96bfdeccb6d92d141b965cb8ba50814": (0, 0, 1),
    # Teapot
    "sem_Teapot_139af10e719186bddaa266217a65f9d7": (0, 0, 1),
    "sem_Teapot_5a471458da447600fea9cf313bd7758b": (0, 0, 1),
    "sem_Teapot_93c6553dc3f2ad11012cc02986a86c3": (0, 0, 1),
    "sem_Teapot_c7a70db33a8c900dab5b523beb03efcd": (0, 0, 1),
    "sem_Teapot_ef16485d8a6750e84212ff51b27f0221": (0, 0, 1),
    # TissueBox
    "sem_TissueBox_13a913b33fb29f7741f9f28bc4fa9522": (0, 0, 1),
    "sem_TissueBox_266911058f9345d8c6e01b363ae6db1b": (0, 0, 1),
    "sem_TissueBox_5128910785490905c58e834f0b160845": (0, 0, 1),
    "sem_TissueBox_670be347dc6411cd3504403c94a7689": (0, 0, 1),
    "sem_TissueBox_734fb9603548dded66c0c88d96ba938f": (0, 0, 1),
    "sem_TissueBox_78f9f10fde0ae0bcd2673f5dfbf4ce1": (0, 0, 1),
    "sem_TissueBox_8808de6ba868499bb8183a4a81361b94": (0, 0, 1),
    "sem_TissueBox_8fb58f8ea50060d9815d24802381b56b": (0, 0, 1),
    # Toaster
    "sem_Toaster_2fb52b43697b9341b785a4ac4a0dbd73": (0, 0, 1),
    "sem_Toaster_3ffbb0ab0f8da32f86271197b958e3d5": (0, 0, 1),
    # ToasterOven
    "sem_ToasterOven_3c357d1352d2d1811fddae104d1cd00e": (0, 0, -1),
    "sem_ToasterOven_95b4eee44e668dfb87a77620548f66e9": (0, 0, -1),
    # ToiletPaper
    "sem_ToiletPaper_3585a9ea9a1c40d5533775ea6714a372": (0, 0, -1),
    "sem_ToiletPaper_39ee1fdc9b314b4c8ad12ca6224f4ace": (0, 0, -1),
    "sem_ToiletPaper_3ae42af85250afde9e40e3e25a634011": (0, -1, 0),
    "sem_ToiletPaper_6658857ea89df65ea35a7666f0cfa5bb": (-1, 0, 0),
    "sem_ToiletPaper_6f97a8c3a5f076665d73afbd7c310e93": (0, 0, 1),
    "sem_ToiletPaper_831fcd293080869ba5afb990da543822": (1, 0, 0),
    # Vase
    "sem_Vase_1a8f9295b44b48895e8c5748ca5ef3ea": (0, 0, 1),
    "sem_Vase_20ed6a41058e6b2659f9892433e1b1f4": (0, 0, 1),
    "sem_Vase_2fc5fff977ac930050b92125e5fcb8ac": (0, 0, 1),
    "sem_Vase_32dc55c3e945384dbc5e533ab711fd24": (0, 0, 1),
    "sem_Vase_38e712afedf758c355db916689d826c1": (0, 0, 1),
    "sem_Vase_3a275e00d69810c62600e861c93ad9cc": (0, 0, 1),
    "sem_Vase_3ca4bd70f299100acd6a36c601ac1ec2": (0, 0, 1),
    "sem_Vase_3dfde41f48617120c7d17f1ea7477fc2": (0, 0, 1),
    "sem_Vase_3e8fd535abd8381c614be109ad41e469": (0, 0, 1),
    "sem_Vase_418003a4a24e2c18265d1076b4b6c5c": (0, 0, -1),
    "sem_Vase_4835bb37072cb5dafc23d64476f7d890": (0, 0, 1),
    "sem_Vase_4cd37983220c5949946c2e1ea428382a": (0, 0, 1),
}

#: The six axis-aligned unit vectors. front_for and origin_for both key off the single non-zero
#: component, so anything else has to be refused rather than approximated.
_AXES = frozenset({(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)})


def _axis(up) -> tuple[int, int, int]:
    try:
        t = tuple(int(v) for v in up)
    except (TypeError, ValueError):
        raise ValueError(f"up must be three numbers, got {up!r}") from None
    if t not in _AXES:
        raise ValueError(f"up must be an axis-aligned unit vector, got {up!r}")
    return t


def front_for(up) -> tuple[int, int, int]:
    """A vector perpendicular to `up`, for scene_synthesizer's `front=`.

    It aligns front and up independently, so a front parallel to up is degenerate and the object
    comes out at whatever angle the library falls back to -- which is the symptom this whole table
    exists to remove. (0,1,0) works for every up except one along Y, which takes (0,0,-1); those
    are the two pairings _object_pose already used.
    """
    return (0, 0, -1) if _axis(up)[1] else (0, 1, 0)


def origin_for(up) -> tuple[str, str, str]:
    """The `origin=` triple that rests the object's world-lowest point on its support.

    origin is given in MESH coordinates and rotated with the asset, so the word has to sit on the
    up AXIS and follow its SIGN: a +Z-up mesh rests on its 'bottom', a -Z-up one on its 'top'.
    Measured, on a vase: keeping the old ('com','bottom','com') while fixing up sinks it 6.1 cm
    into the countertop, and 'bottom' on a Z-down mesh hangs it 12.6 cm underneath.
    """
    t = _axis(up)
    i = next(k for k in range(3) if t[k])
    word = "bottom" if t[i] > 0 else "top"
    return tuple(word if k == i else "com" for k in range(3))


def up_for(mesh_path):
    """The recorded up vector for a mesh path, or None when it has no override.

    The name is derived exactly as isaaclab.simvla.utils.load_grasp_file derives it, via
    grasp_manifest.object_name -- three copies of that rule would eventually disagree, and then
    this table would describe one mesh while the scene placed another.
    """
    if not mesh_path:
        return None
    from .grasp_manifest import object_name
    return MESH_UP.get(object_name(mesh_path))


#: Object types that stand on +Z when no mesh entry applies. Everything else takes the core_*
#: convention below -- these two are the exception _object_pose has always carried.
TYPE_UP: dict[str, tuple[int, int, int]] = {"apple": (0, 0, 1), "sodacan": (0, 0, 1)}

#: What a mesh with neither an entry nor a listed type gets: +Y up, the ShapeNetCore convention
#: the core_*/ddg_* meshes are stored in.
FALLBACK_UP: tuple[int, int, int] = (0, 1, 0)


def resolved_up(obj_type, mesh_path=None) -> tuple[int, int, int]:
    """The up vector this (type, mesh) pair will actually be placed with. Never None.

    The single copy of that resolution, because two modules need the same answer and they used to
    hold half of it each: kitchen_build to stand the object on the counter, and kitchen_gallery to
    draw the tile the object is CHOSEN from. While only the placer applied it, every picker tile
    was the raw file frame -- so a core_* bottle lay on its side in the gallery and then stood up
    in the scene, and an unlabelled sem_* cereal box stood up in the gallery and was then laid
    down. Same function on both sides means the tile cannot disagree with the counter.
    """
    up = up_for(mesh_path)
    if up is None:
        up = TYPE_UP.get(obj_type, FALLBACK_UP)
    return tuple(up)


#: Labelled upright already, so deliberately NOT overridden -- kept as a record that the first
#: pass flagged them and a closer look cleared them.
CONFIRMED_DEFAULT = ('core_jar_2886e2026ed9da084b3c42e318f3affc',)

from typing import Dict, List, Tuple, Set, Any, Optional
import math
from collections import defaultdict


def f_beta(pred_list: list[str], true_list: list[str], beta=0.5, eps=1e-8) -> float:
    pred_set = set(pred_list)
    true_set = set(true_list)
    tp = len(pred_set & true_set)
    P = tp / (len(pred_list) + eps)
    R = tp / (len(true_list) + eps)
    b2 = beta * beta
    return (1 + b2) * P * R / (b2 * P + R + eps)

# thresh 为不扣分的阈值，30 为最大容忍阈值-thresh
def len_penalty(n: int, thresh=30, lam_quad=1.0/(30**2)) -> float:
    d = max(0, n - thresh)
    lam_quad = 1.0/(thresh**2)
    return min(lam_quad * (d ** 2),1)

def len_penalty_linear(n: int, thresh=30, lam_lin=1.0/30) -> float:
    d = max(0, n - thresh)
    lam_lin = 1.0 / thresh
    return min(lam_lin * d, 1.0)


def section_reward_from_refs(pred_refs: List[str], true_refs: List[str],
                             beta=0.5, Lmax=30 ) -> Dict[str, Any]:
    """
    单章节奖励（matched）
    返回 dict 包含 precision, recall, f_beta, len_penalty, reward
    reward = f_beta - len_penalty(len(pred_refs))
    """
    f = f_beta(pred_refs, true_refs, beta)
    pen = len_penalty(len(pred_refs), Lmax)
    reward = f - pen
    # clip reward 可以考虑但按公式不强制裁剪
    return {
        "precision_recall_fbeta": None,  # 保留位置以便扩展
        "f_beta": f,
        "len_pen": pen,
        "reward": reward,
        "pred_len": len(pred_refs),
        "true_len": len(true_refs)
    }


def make_section_map(sections: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    将 section 列表 转换成 id -> ref_list 映射
    id 的决定顺序： 'numbering' > 'title' > index
    每项确保返回 list[str] （若没有 'ref' 或为 None 则视为空列表）
    """
    mapping: Dict[str, List[str]] = {}
    for i, sec in enumerate(sections):
        if isinstance(sec, dict):
            if 'numbering' in sec and sec.get('numbering') not in (None, ""):
                sid = f"num:{sec['numbering']}"
            elif 'title' in sec and sec.get('title') not in (None, ""):
                sid = f"title:{sec['title']}"
            else:
                sid = f"idx:{i}"
            refs = sec.get('ref', []) or []
            # 保证全是字符串
            refs = [str(r) for r in refs]
            mapping[sid] = refs
        else:
            # 非 dict 的项，按索引命名并尝试转成 list[str]
            sid = f"idx:{i}"
            try:
                refs = list(sec)
                refs = [str(r) for r in refs]
            except Exception:
                refs = []
            mapping[sid] = refs
    return mapping

def greedy_match_by_numbering(gen_map: Dict[str, List[str]],
                              true_map: Dict[str, List[str]]
                              ) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """
    按 numbering 字段 (我们在 make_section_map 中用 'num:<numbering>' 作为 key) 进行一对一匹配。
    - 如果同一个 numbering 在两边都有多条记录，则按出现顺序一一配对（最多 min(len(gen_list), len(true_list)) 对）。
    - 不含 numbering 的条目（如 'title:' 或 'idx:' 开头的 key）不会被匹配，归为 unmatched。
    返回 (pairs, unmatched_true_ids, unmatched_gen_ids)
    """
    # build numbering -> list of ids
    gen_by_num = defaultdict(list)   # num -> [gen_id, ...]
    true_by_num = defaultdict(list)  # num -> [true_id, ...]
    # also collect all ids
    all_gen_ids = list(gen_map.keys())
    all_true_ids = list(true_map.keys())

    # helper to extract numbering from key built by make_section_map
    def extract_numbering(key: str) -> str:
        if key.startswith("num:"):
            return key[len("num:"):]
        return None

    for gid in all_gen_ids:
        num = extract_numbering(gid)
        gen_by_num[num].append(gid)

    for tid in all_true_ids:
        num = extract_numbering(tid)
        true_by_num[num].append(tid)

    pairs: List[Tuple[str, str]] = []
    used_gen = set()
    used_true = set()

    # 对存在于两边的 numbering 做一对一配对
    common_nums = set(gen_by_num.keys()) & set(true_by_num.keys())
    # 排除 None（即无 numbering 的条目），因为我们不按照 title/index 匹配
    if None in common_nums:
        common_nums.remove(None)

    for num in common_nums:
        gen_list = gen_by_num[num]
        true_list = true_by_num[num]
        m = min(len(gen_list), len(true_list))
        for i in range(m):
            gid = gen_list[i]
            tid = true_list[i]
            pairs.append((gid, tid))
            used_gen.add(gid)
            used_true.add(tid)

    # unmatched：那些没有被配到的一方全部视为 unmatched
    unmatched_true = [tid for tid in all_true_ids if tid not in used_true]
    unmatched_gen = [gid for gid in all_gen_ids if gid not in used_gen]

    return pairs, unmatched_true, unmatched_gen

def article_reward(gen_sections: List[Dict[str, Any]],
                   true_sections: List[Dict[str, Any]],
                   beta=0.5,
                   Lmax: Optional[int] = None,
                   lam: Optional[float] = None,
                   quadratic: bool = True,
                   average: bool = True,
                   return_details: bool = True) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    计算整篇文章的奖励：
      gen_sections, true_sections: List[Dict]（每个 dict 含 'ref' 字段）
    参数：
      - beta: F_beta 的 beta
      - Lmax: 扣分阈值（若为 None，则取 true_sections 中 ref 长度的最大值，若 M==0 则默认 30）
      - lam: 惩罚系数（若为 None 会根据 Lmax 自动设定）
      - quadratic: True 使用二次惩罚，否则线性
      - average: 是否按真实章节数 N 归一化（题目要求按 N）
      - return_details: 是否返回每章节明细
    返回:
      (R_article, details_dict or None)
    """
    gen_map = make_section_map(gen_sections)
    true_map = make_section_map(true_sections)
    true_norefs = 0
    # 确定 Lmax , 规定 Lmax 最多为 10
    if Lmax is None:
        # 以 true 中最大 ref 长度为阈值（如题中建议）
        max_true_len = 0
        for refs in true_map.values():
            max_true_len = max(max_true_len, len(refs))
            if len(refs) == 0:
                true_norefs = true_norefs+1
        Lmax = max_true_len if max_true_len > 0 else 30
        #print(f"Lmax{Lmax}")
    Lmax = max(Lmax, 10)
    # 计算 num 级的匹配度
    pairs, missed_true, spurious_gen = greedy_match_by_numbering(gen_map, true_map)

    N_true = max(1, len(true_map)-true_norefs)  # 避免除零；但题中 N 为真实章节数
    total = 0.0
    details = {
        "matched": [],
        "missed": [],
        "spurious": [],
        "N_true": len(true_map),
        "N_gen": len(gen_map)
    }

    # 已匹配章节：R_i = Fbeta(pred_i,true_i) - LenPen(|pred_i|)
    for gid, tid in pairs:
        pred_refs = gen_map.get(gid, [])
        true_refs = true_map.get(tid, [])
        sec = section_reward_from_refs(pred_refs, true_refs, beta, Lmax)
        total += sec["reward"]
        details["matched"].append({
            "gen_id": gid, "true_id": tid,
            "pred_len": sec["pred_len"], "true_len": sec["true_len"],
            "f_beta": sec["f_beta"], "len_pen": sec["len_pen"], "reward": sec["reward"]
        })

    # 漏章节惩罚： R_miss = -LenPen(|true_i|)
    for tid in missed_true:
        true_refs = true_map.get(tid, [])
        pen = len_penalty(len(true_refs), Lmax)
        r = -pen
        total += r
        details["missed"].append({"true_id": tid, "true_len": len(true_refs), "len_pen": pen, "reward": r})

    # 虚假章节惩罚： R_spurious = -LenPen(|pred_i|)
    for gid in spurious_gen:
        pred_refs = gen_map.get(gid, [])
        pen = len_penalty(len(pred_refs), Lmax)
        r = -pen
        total += r
        details["spurious"].append({"gen_id": gid, "pred_len": len(pred_refs), "len_pen": pen, "reward": r})

    R_article = (total / N_true) if average else total
    R_article = max(-1,R_article)
    if return_details:
        details["R_article"] = R_article
        details["total_raward"] = total
        return R_article, details
    else:
        return R_article, None

# -------------------------
# 示例（可直接运行检验）
# -------------------------
if __name__ == "__main__":
    # 非常极端的示例
    # 一个简单的示例：pred 和 true 完全相同
    pred = [
        {'level': 1, 'numbering': '1', 'title': 'Introduction', 'ref': []}, {'level': 2, 'numbering': '1.1', 'title': 'Approaches to Synthesis', 'ref': []}, {'level': 1, 'numbering': '2', 'title': 'Developments in Oxide Molecular Beam Epitaxy', 'ref': ['bednorz_possible_1986', 'jin_thousandfold_1994', 'wang_epitaxial_2003', 'matsuno_interface-driven_2016', 'brahlek_frontiers_2018', 'schlom_perspective_2015', 'osti_1616509', 'das_sarma_proposal_2006', 'cava_introduction_2021', 'witczak-krempa_correlated_2014', 'christen_recent_2008', 'ohtomo_high-mobility_2004', 'breckenfeld_effect_2013', 'breckenfeld_effects_2014', 'li_superconductivity_2019', 'haeni_rheed_2000', 'mckee_crystalline_1998', 'bozovic_atomic-level_1997', 'kumah_effect_2014', 'disa_orbital_2015', 'may_enhanced_2009', 'bhattacharya_metal-insulator_2008', 'monkman_quantum_2012', 'ihlefeld_adsorption-controlled_2007', 'scafetta_band_2014', 'comes_interface_2016', 'caspi_effect_2022', 'lee_strong_2019', 'andersen_layer-by-layer_2018', 'chambers_band_2011', 'comes_probing_2017', 'lin_interface_2018', 'al-tawhid_two-dimensional_2021', 'qiao_epitaxial_2011', 'warusawithana_laalo3_2013', 'segal_x-ray_2009', 'chambers_instability_2010', 'nair_synthesis_2018', 'nie_interplay_2015', 'du_self-corrected_2014', 'provence_machine_2020', 'vasudevan_big-data_2014', 'suyolcu_engineering_2022', 'wakabayashi_machine-learning-assisted_2019']}, {'level': 2, 'numbering': '2.1', 'title': 'Hybrid and Metalorganic MBE', 'ref': []}, {'level': 2, 'numbering': '2.2', 'title': 'Suboxide MBE', 'ref': ['bednorz_possible_1986', 'jin_thousandfold_1994', 'wang_epitaxial_2003', 'matsuno_interface-driven_2016', 'brahlek_frontiers_2018', 'schlom_perspective_2015']}, {'level': 2, 'numbering': '2.3', 'title': 'Thermal Laser MBE', 'ref': []}, {'level': 2, 'numbering': '2.4', 'title': 'Other Developments', 'ref': ['christen_recent_2008', 'wang_epitaxial_2003', 'ohtomo_high-mobility_2004', 'breckenfeld_effect_2013', 'breckenfeld_effects_2014', 'li_superconductivity_2019', 'haeni_rheed_2000', 'mckee_crystalline_1998', 'bozovic_atomic-level_1997', 'kumah_effect_2014', 'disa_orbital_2015', 'may_enhanced_2009', 'bhattacharya_metal-insulator_2008', 'monkman_quantum_2012', 'ihlefeld_adsorption-controlled_2007', 'scafetta_band_2014', 'comes_interface_2016', 'caspi_effect_2022', 'lee_strong_2019', 'andersen_layer-by-layer_2018', 'chambers_band_2011', 'comes_probing_2017', 'lin_interface_2018', 'al-tawhid_two-dimensional_2021', 'qiao_epitaxial_2011', 'warusawithana_laalo3_2013', 'segal_x-ray_2009', 'chambers_instability_2010', 'nair_synthesis_2018', 'nie_interplay_2015', 'du_self-corrected_2014', 'provence_machine_2020', 'vasudevan_big-data_2014', 'suyolcu_engineering_2022', 'wakabayashi_machine-learning-assisted_2019']}, {'level': 1, 'numbering': '3', 'title': 'Magnetic and Ferroelectric Oxides', 'ref': []}, {'level': 1, 'numbering': '4', 'title': 'Superconducting Oxides', 'ref': ['kuech_recent_1992', 'brahlek_frontiers_2018', 'jalan_growth_2009', 'jalan_molecular_2009', 'son_epitaxial_2010', 'xu_stoichiometry-driven_2014', 'xu_predictive_2016', 'moetakef_growth_2012', 'eaton_growth_2015', 'brahlek_accessing_2015', 'prakash_adsorption-controlled_2017', 'prakash_wide_2017', 'thapa_surface_2022', 'thapa_correlating_2021', 'nunn_novel_2021', 'nunn_solid-source_2021', 'nair_engineering_2023', 'vogt_adsorption-controlled_2021', 'adkison_suitability_2020', 'du_iso-oriented_2016', 'li_crystallographic_2015', 'schwaigert_molecular_2023', 'kuznetsova_growth_2023', 'raghavan_high-mobility_2016', 'hensling_epitaxial_2022', 'Vogt2022', 'smart_thermal_2021', 'braun_film_2019', 'kim_thermal_2021', 'kim_thermal_2022', 'braun_situ_2020', 'rimal_diffusion-assisted_2022', 'Kim2023_arxiv', 'Kum2020', 'kim_remote_2022', 'yoon_freestanding_2022', 'Ying2022', 'Chambers2000a', 'Mannhart2010', 'Ramesh2019', 'Bhattacharya2014', 'Schooley1964', 'Sleight1975', 'Bednorz1986', 'Wu1987']}, {'level': 2, 'numbering': '4.1', 'title': 'Cuprates: The original high T$_c$ superconductor', 'ref': ['jalan_molecular_2009', 'son_epitaxial_2010', 'brahlek_frontiers_2018', 'thapa_correlating_2021', 'brahlek_accessing_2015', 'prakash_adsorption-controlled_2017']}, {'level': 2, 'numbering': '4.2', 'title': 'Nickelates: Isoelectronic analogue to the cuprates', 'ref': ['thapa_surface_2022', 'nunn_novel_2021', 'nunn_solid-source_2021', 'nair_engineering_2023', 'vogt_adsorption-controlled_2021', 'adkison_suitability_2020', 'du_iso-oriented_2016', 'li_crystallographic_2015', 'schwaigert_molecular_2023', 'kuznetsova_growth_2023', 'raghavan_high-mobility_2016', 'hensling_epitaxial_2022', 'Vogt2022', 'smart_thermal_2021', 'braun_film_2019', 'kim_thermal_2021', 'kim_thermal_2022', 'braun_situ_2020', 'rimal_diffusion-assisted_2022', 'Kim2023_arxiv']}, {'level': 2, 'numbering': '4.3', 'title': 'Tantalates: A 5d superconductor', 'ref': ['Kum2020', 'kim_remote_2022', 'yoon_freestanding_2022', 'Ying2022']}, {'level': 2, 'numbering': '4.4', 'title': 'Non transition metal oxides', 'ref': ['Chambers2000a', 'Mannhart2010', 'Ramesh2019', 'Bhattacharya2014', 'Schooley1964', 'Sleight1975', 'Bednorz1986', 'Wu1987']}, {'level': 1, 'numbering': '5', 'title': 'Topological Phenomena in Oxides', 'ref': ['Webb1987', 'baiutti_oxide_2018', 'yamamoto_epitaxial_2015', 'Bozovic2001', 'suyolcu_octahedral_2017', 'suyolcu_a-axis_2021', 'bonmassar_design_2023', 'krockenberger_infinite-layer_2018', 'Harter2015', 'Zhong2018', 'Logvenov2009', 'Logvenov2013', 'Bozovic2016', 'suyolcu_design_2020', 'xu_synthesis_2022', 'haenel_incoherent_2022', 'tummuru_josephson_2022', 'margalit_chiral_2022', 'song_doping_2022', 'zhao2021emergent', 'catalano_rare-earth_2018', 'king_atomic-scale_2014', 'provence_machine_2020', 'may_onset_2009', 'li_self-healing_2022', 'yan_situ_2020', 'kumah_effect_2014', 'li_superconductivity_2019', 'nomura_superconductivity_2022', 'bernardini_thin-film_2022', 'ding_critical_2023', 'pan_superconductivity_2022', 'ferenc_segedin_limits_2023', 'pan_synthesis_2022', 'wei_superconducting_2023', 'wei_solid_2023', 'ueno_discovery_2011', 'Liu2021', 'liu_tunable_2023', 'arnault_anisotropic_2023', 'al-tawhid_enhanced_2023', 'gupta_ktao3new_2022', 'schwaigert_molecular_2023', 'Sleight1975', 'batlogg_superconductivity_1989', 'mattheiss_superconductivity_1988', 'cava_superconductivity_1988', 'hellman_molecular_1989', 'bozovic_quest_2020', 'yan_large-energy-gap_2013', 'harris_superconductivity-localization_2018', 'harris_charge_2020', 'sleight_bismuthates_2015', 'kim_superconductivity_2022', 'avron_topological_2003', 'Murakami2004', 'Konig2012', 'Bansil2016', 'Narang2020', 'Witczak-Krempa2014', 'Vergniory2019', 'topologicalQuantumChemistry']}, {'level': 2, 'numbering': '5.1', 'title': 'Ruthenates: Topological textures and superconductivity', 'ref': ['catalano_rare-earth_2018', 'king_atomic-scale_2014', 'provence_machine_2020', 'may_onset_2009', 'li_self-healing_2022', 'yan_situ_2020', 'kumah_effect_2014', 'li_superconductivity_2019', 'nomura_superconductivity_2022', 'bernardini_thin-film_2022', 'ding_critical_2023', 'pan_superconductivity_2022', 'ferenc_segedin_limits_2023', 'pan_synthesis_2022', 'wei_superconducting_2023', 'wei_solid_2023']}, {'level': 2, 'numbering': '5.2', 'title': 'Iridates: Dirac and Weyl materials', 'ref': ['ueno_discovery_2011', 'Liu2021', 'liu_tunable_2023', 'arnault_anisotropic_2023', 'al-tawhid_enhanced_2023', 'gupta_ktao3new_2022', 'schwaigert_molecular_2023', 'Sleight1975', 'batlogg_superconductivity_1989', 'mattheiss_superconductivity_1988', 'cava_superconductivity_1988', 'hellman_molecular_1989', 'bozovic_quest_2020', 'yan_large-energy-gap_2013', 'harris_superconductivity-localization_2018', 'harris_charge_2020', 'sleight_bismuthates_2015', 'kim_superconductivity_2022', 'avron_topological_2003', 'Murakami2004', 'Konig2012', 'Bansil2016', 'Narang2020', 'Witczak-Krempa2014', 'Vergniory2019', 'topologicalQuantumChemistry']}, {'level': 1, 'numbering': '6', 'title': 'Future Directions and Conclusion', 'ref': ['He2022', 'Junquera2023', 'tokura_magnetic_2021', 'kuepferling_measuring_2023', 'Ohuchi2015', 'Yun2018', 'ohuchi_electric-field_2018', 'Sohn_stable_2021', 'Kimbell2020', 'Kim2020', 'Skoropata2021', 'Wang2020', 'schreiber_model_2023', 'manjeshwar_adsorption-controlled_2023', 'nair_synthesis_2018', 'gu_overview_2022']}
        ]
    true = [
        {'level': 1, 'numbering': '1', 'title': 'Introduction', 'ref': []}, {'level': 2, 'numbering': '1.1', 'title': 'Approaches to Synthesis', 'ref': []}, {'level': 1, 'numbering': '2', 'title': 'Developments in Oxide Molecular Beam Epitaxy', 'ref': ['bednorz_possible_1986', 'jin_thousandfold_1994', 'wang_epitaxial_2003', 'matsuno_interface-driven_2016', 'brahlek_frontiers_2018', 'schlom_perspective_2015', 'osti_1616509', 'das_sarma_proposal_2006', 'cava_introduction_2021', 'witczak-krempa_correlated_2014', 'christen_recent_2008', 'ohtomo_high-mobility_2004', 'breckenfeld_effect_2013', 'breckenfeld_effects_2014', 'li_superconductivity_2019', 'haeni_rheed_2000', 'mckee_crystalline_1998', 'bozovic_atomic-level_1997', 'kumah_effect_2014', 'disa_orbital_2015', 'may_enhanced_2009', 'bhattacharya_metal-insulator_2008', 'monkman_quantum_2012', 'ihlefeld_adsorption-controlled_2007', 'scafetta_band_2014', 'comes_interface_2016', 'caspi_effect_2022', 'lee_strong_2019', 'andersen_layer-by-layer_2018', 'chambers_band_2011', 'comes_probing_2017', 'lin_interface_2018', 'al-tawhid_two-dimensional_2021', 'qiao_epitaxial_2011', 'warusawithana_laalo3_2013', 'segal_x-ray_2009', 'chambers_instability_2010', 'nair_synthesis_2018', 'nie_interplay_2015', 'du_self-corrected_2014', 'provence_machine_2020', 'vasudevan_big-data_2014', 'suyolcu_engineering_2022', 'wakabayashi_machine-learning-assisted_2019']}, {'level': 2, 'numbering': '2.1', 'title': 'Hybrid and Metalorganic MBE', 'ref': []}, {'level': 2, 'numbering': '2.2', 'title': 'Suboxide MBE', 'ref': ['bednorz_possible_1986', 'jin_thousandfold_1994', 'wang_epitaxial_2003', 'matsuno_interface-driven_2016', 'brahlek_frontiers_2018', 'schlom_perspective_2015']}, {'level': 2, 'numbering': '2.3', 'title': 'Thermal Laser MBE', 'ref': []}, {'level': 2, 'numbering': '2.4', 'title': 'Other Developments', 'ref': ['christen_recent_2008', 'wang_epitaxial_2003', 'ohtomo_high-mobility_2004', 'breckenfeld_effect_2013', 'breckenfeld_effects_2014', 'li_superconductivity_2019', 'haeni_rheed_2000', 'mckee_crystalline_1998', 'bozovic_atomic-level_1997', 'kumah_effect_2014', 'disa_orbital_2015', 'may_enhanced_2009', 'bhattacharya_metal-insulator_2008', 'monkman_quantum_2012', 'ihlefeld_adsorption-controlled_2007', 'scafetta_band_2014', 'comes_interface_2016', 'caspi_effect_2022', 'lee_strong_2019', 'andersen_layer-by-layer_2018', 'chambers_band_2011', 'comes_probing_2017', 'lin_interface_2018', 'al-tawhid_two-dimensional_2021', 'qiao_epitaxial_2011', 'warusawithana_laalo3_2013', 'segal_x-ray_2009', 'chambers_instability_2010', 'nair_synthesis_2018', 'nie_interplay_2015', 'du_self-corrected_2014', 'provence_machine_2020', 'vasudevan_big-data_2014', 'suyolcu_engineering_2022', 'wakabayashi_machine-learning-assisted_2019']}, {'level': 1, 'numbering': '3', 'title': 'Magnetic and Ferroelectric Oxides', 'ref': []}, {'level': 1, 'numbering': '4', 'title': 'Superconducting Oxides', 'ref': ['kuech_recent_1992', 'brahlek_frontiers_2018', 'jalan_growth_2009', 'jalan_molecular_2009', 'son_epitaxial_2010', 'xu_stoichiometry-driven_2014', 'xu_predictive_2016', 'moetakef_growth_2012', 'eaton_growth_2015', 'brahlek_accessing_2015', 'prakash_adsorption-controlled_2017', 'prakash_wide_2017', 'thapa_surface_2022', 'thapa_correlating_2021', 'nunn_novel_2021', 'nunn_solid-source_2021', 'nair_engineering_2023', 'vogt_adsorption-controlled_2021', 'adkison_suitability_2020', 'du_iso-oriented_2016', 'li_crystallographic_2015', 'schwaigert_molecular_2023', 'kuznetsova_growth_2023', 'raghavan_high-mobility_2016', 'hensling_epitaxial_2022', 'Vogt2022', 'smart_thermal_2021', 'braun_film_2019', 'kim_thermal_2021', 'kim_thermal_2022', 'braun_situ_2020', 'rimal_diffusion-assisted_2022', 'Kim2023_arxiv', 'Kum2020', 'kim_remote_2022', 'yoon_freestanding_2022', 'Ying2022', 'Chambers2000a', 'Mannhart2010', 'Ramesh2019', 'Bhattacharya2014', 'Schooley1964', 'Sleight1975', 'Bednorz1986', 'Wu1987']}, {'level': 2, 'numbering': '4.1', 'title': 'Cuprates: The original high T$_c$ superconductor', 'ref': ['jalan_molecular_2009', 'son_epitaxial_2010', 'brahlek_frontiers_2018', 'thapa_correlating_2021', 'brahlek_accessing_2015', 'prakash_adsorption-controlled_2017']}, {'level': 2, 'numbering': '4.2', 'title': 'Nickelates: Isoelectronic analogue to the cuprates', 'ref': ['thapa_surface_2022', 'nunn_novel_2021', 'nunn_solid-source_2021', 'nair_engineering_2023', 'vogt_adsorption-controlled_2021', 'adkison_suitability_2020', 'du_iso-oriented_2016', 'li_crystallographic_2015', 'schwaigert_molecular_2023', 'kuznetsova_growth_2023', 'raghavan_high-mobility_2016', 'hensling_epitaxial_2022', 'Vogt2022', 'smart_thermal_2021', 'braun_film_2019', 'kim_thermal_2021', 'kim_thermal_2022', 'braun_situ_2020', 'rimal_diffusion-assisted_2022', 'Kim2023_arxiv']}, {'level': 2, 'numbering': '4.3', 'title': 'Tantalates: A 5d superconductor', 'ref': ['Kum2020', 'kim_remote_2022', 'yoon_freestanding_2022', 'Ying2022']}, {'level': 2, 'numbering': '4.4', 'title': 'Non transition metal oxides', 'ref': ['Chambers2000a', 'Mannhart2010', 'Ramesh2019', 'Bhattacharya2014', 'Schooley1964', 'Sleight1975', 'Bednorz1986', 'Wu1987']}, {'level': 1, 'numbering': '5', 'title': 'Topological Phenomena in Oxides', 'ref': ['Webb1987', 'baiutti_oxide_2018', 'yamamoto_epitaxial_2015', 'Bozovic2001', 'suyolcu_octahedral_2017', 'suyolcu_a-axis_2021', 'bonmassar_design_2023', 'krockenberger_infinite-layer_2018', 'Harter2015', 'Zhong2018', 'Logvenov2009', 'Logvenov2013', 'Bozovic2016', 'suyolcu_design_2020', 'xu_synthesis_2022', 'haenel_incoherent_2022', 'tummuru_josephson_2022', 'margalit_chiral_2022', 'song_doping_2022', 'zhao2021emergent', 'catalano_rare-earth_2018', 'king_atomic-scale_2014', 'provence_machine_2020', 'may_onset_2009', 'li_self-healing_2022', 'yan_situ_2020', 'kumah_effect_2014', 'li_superconductivity_2019', 'nomura_superconductivity_2022', 'bernardini_thin-film_2022', 'ding_critical_2023', 'pan_superconductivity_2022', 'ferenc_segedin_limits_2023', 'pan_synthesis_2022', 'wei_superconducting_2023', 'wei_solid_2023', 'ueno_discovery_2011', 'Liu2021', 'liu_tunable_2023', 'arnault_anisotropic_2023', 'al-tawhid_enhanced_2023', 'gupta_ktao3new_2022', 'schwaigert_molecular_2023', 'Sleight1975', 'batlogg_superconductivity_1989', 'mattheiss_superconductivity_1988', 'cava_superconductivity_1988', 'hellman_molecular_1989', 'bozovic_quest_2020', 'yan_large-energy-gap_2013', 'harris_superconductivity-localization_2018', 'harris_charge_2020', 'sleight_bismuthates_2015', 'kim_superconductivity_2022', 'avron_topological_2003', 'Murakami2004', 'Konig2012', 'Bansil2016', 'Narang2020', 'Witczak-Krempa2014', 'Vergniory2019', 'topologicalQuantumChemistry']}, {'level': 2, 'numbering': '5.1', 'title': 'Ruthenates: Topological textures and superconductivity', 'ref': ['catalano_rare-earth_2018', 'king_atomic-scale_2014', 'provence_machine_2020', 'may_onset_2009', 'li_self-healing_2022', 'yan_situ_2020', 'kumah_effect_2014', 'li_superconductivity_2019', 'nomura_superconductivity_2022', 'bernardini_thin-film_2022', 'ding_critical_2023', 'pan_superconductivity_2022', 'ferenc_segedin_limits_2023', 'pan_synthesis_2022', 'wei_superconducting_2023', 'wei_solid_2023']}, {'level': 2, 'numbering': '5.2', 'title': 'Iridates: Dirac and Weyl materials', 'ref': ['ueno_discovery_2011', 'Liu2021', 'liu_tunable_2023', 'arnault_anisotropic_2023', 'al-tawhid_enhanced_2023', 'gupta_ktao3new_2022', 'schwaigert_molecular_2023', 'Sleight1975', 'batlogg_superconductivity_1989', 'mattheiss_superconductivity_1988', 'cava_superconductivity_1988', 'hellman_molecular_1989', 'bozovic_quest_2020', 'yan_large-energy-gap_2013', 'harris_superconductivity-localization_2018', 'harris_charge_2020', 'sleight_bismuthates_2015', 'kim_superconductivity_2022', 'avron_topological_2003', 'Murakami2004', 'Konig2012', 'Bansil2016', 'Narang2020', 'Witczak-Krempa2014', 'Vergniory2019', 'topologicalQuantumChemistry']}, {'level': 1, 'numbering': '6', 'title': 'Future Directions and Conclusion', 'ref': ['He2022', 'Junquera2023', 'tokura_magnetic_2021', 'kuepferling_measuring_2023', 'Ohuchi2015', 'Yun2018', 'ohuchi_electric-field_2018', 'Sohn_stable_2021', 'Kimbell2020', 'Kim2020', 'Skoropata2021', 'Wang2020', 'schreiber_model_2023', 'manjeshwar_adsorption-controlled_2023', 'nair_synthesis_2018', 'gu_overview_2022']}
        ]

    score, det = article_reward(pred, true, beta=0.5, Lmax=None, quadratic=True, return_details=True)
    print("R_article:", score)
    import json
    print(json.dumps(det, indent=2, ensure_ascii=False))
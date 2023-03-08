"""Definitions of functional groups using SMARTS strings."""

# Define SMARTS
alkane = '[CX4;H3,H2]'
alkene = '[CX3]=[CX3]'
alkyne = '[CX2]#[CX2]'
arene = '[cX3]1[cX3][cX3][cX3][cX3][cX3]1'
haloalkane = '[#6][F,Cl,Br,I]'
alcohol = '[#6][OX2H]'
aldehyde = '[CX3H1](=O)[#6,H]'
ketone = '[#6][CX3](=O)[#6]'
carboxylic_acid = '[CX3](=O)[OX2H]'
acid_anhydride = '[CX3](=[OX1])[OX2][CX3](=[OX1])'
acyl_halide = '[CX3](=[OX1])[F,Cl,Br,I]'
ester = '[#6][CX3](=O)[OX2H0][#6]'
ether = '[OD2]([#6])[#6]'
amine = '[NX3;H2,H1,H0;!$(NC=O)]'
amide = '[NX3][CX3](=[OX1])[#6]'
nitrile = '[NX1]#[CX2]'
imide = '[CX3](=[OX1])[NX3][CX3](=[OX1])'
imine = '[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]'
azo_compound = '[#6][NX2]=[NX2][#6]'
thiol = '[#16X2H]'
thial = '[CX3H1](=O)[#6,H]'
sulfone = '[#16X4](=[OX1])(=[OX1])([#6])[#6]'
sulfonic_acid = '[#16X4](=[OX1])(=[OX1])([#6])[OX2H]'
enol = '[OX2H][#6X3]=[#6]'
phenol = '[OX2H][cX3]:[c]'
hydrazine = '[NX3][NX3]'
enamine = '[NX3][CX3]=[CX3]'
isocyanate = '[NX2]=[C]=[O]'
isothiocyanate = '[NX2]=[C]=[S]'
phosphine = '[PX3]'
sulfonamide = '[#16X4]([NX3])(=[OX1])(=[OX1])[#6]'
sulfonate = '[#16X4](=[OX1])(=[OX1])([#6])[OX2H0]'
sulfoxide = '[#16X3]=[OX1]'
thioamide = '[NX3][CX3]=[SX1]'
hydrazone = '[NX3][NX2]=[#6]'
carbamate = '[NX3][CX3](=[OX1])[OX2H0]'
sulfide = '[#16X2H0]'


fg_list_extended = [
        alkane,
        alkene,
        alkyne,
        arene, 
        haloalkane, 
        alcohol,
        aldehyde,
        ketone, 
        carboxylic_acid, 
        acid_anhydride, 
        acyl_halide, 
        ester, 
        ether, 
        amine, 
        amide, 
        nitrile, 
        imide, 
        imine, 
        azo_compound,
        thiol, 
        thial, 
        sulfone, 
        sulfonic_acid, 
        enol, 
        phenol,
        hydrazine, 
        enamine, 
        isocyanate, 
        isothiocyanate,
        phosphine, 
        sulfonamide, 
        sulfonate, 
        sulfoxide, 
        thioamide, 
        hydrazone, 
        carbamate, 
        sulfide]


fg_list_original = [
        alkane,
        alkene,
        alkyne,
        arene, 
        haloalkane, 
        alcohol,
        aldehyde,
        ketone, 
        carboxylic_acid, 
        acid_anhydride, 
        acyl_halide, 
        ester, 
        ether, 
        amine, 
        amide, 
        nitrile, 
        imide, 
        imine, 
        azo_compound,
        thiol, 
        thial,
        phenol]


label_names_extended = [
        'Alkane',
        'Alkene',
        'Alkyne',
        'Arene', 
        'Haloalkane', 
        'Alcohol',
        'Aldehyde',
        'Ketone', 
        'Carboxylic acid', 
        'Acid anhydride', 
        'Acyl halide', 
        'Ester', 
        'Ether', 
        'Amine', 
        'Amide', 
        'Nitrile', 
        'Imide', 
        'Imine', 
        'Azo compound',
        'Thiol', 
        'Thial', 
        'Sulfone', 
        'Sulfonic acid', 
        'Enol', 
        'Phenol',
        'Hydrazine', 
        'Enamine', 
        'Isocyanate', 
        'Isothiocyanate', 
        'Phosphine', 
        'Sulfonamide', 
        'Sulfonate', 
        'Sulfoxide', 
        'Thioamide', 
        'Hydrazone', 
        'Carbamate', 
        'Sulfide']


label_names_original = [
        'Alkane',
        'Alkene',
        'Alkyne',
        'Arene', 
        'Haloalkane', 
        'Alcohol',
        'Aldehyde',
        'Ketone', 
        'Carboxylic acid', 
        'Acid anhydride', 
        'Acyl halide', 
        'Ester', 
        'Ether', 
        'Amine', 
        'Amide', 
        'Nitrile', 
        'Imide', 
        'Imine', 
        'Azo compound',
        'Thiol', 
        'Thial',
        'Phenol']
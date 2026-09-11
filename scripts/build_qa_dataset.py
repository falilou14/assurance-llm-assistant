"""
Reconstruit data/processed/dataset.jsonl avec de VRAIES paires question/réponse
ancrées dans data/raw/*.txt -- contrairement à la v1 (build_instruction dans
dataset_builder.py), qui répétait la même instruction fixe pour les 15 chunks.

Pourquoi ce changement : le modèle entraîné sur la v1 avait appris un style
("dossier assurance") plutôt qu'une vraie capacité à répondre à une question
précise en s'appuyant sur un contexte -- diagnostiqué via un test réel qui a
produit une hallucination complète (voir le dossier d'étude, session 2).

Chaque exemple ici a :
- instruction : une question variée (formulation naturelle, pas de gabarit fixe)
- input       : le contexte source pertinent (ce que le RAG retrouverait)
- output      : une réponse concise, ancrée dans ce contexte

Deux exemples volontairement hors-sujet / sans contexte sont inclus pour
apprendre au modèle à décliner poliment plutôt qu'inventer.

Usage:
    python scripts/build_qa_dataset.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.dataset_builder import DatasetBuilder

OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "processed", "dataset.jsonl")

EXAMPLES = [
    # --- assurance_auto.txt ---
    {
        "instruction": "Qu'est-ce que couvre une assurance auto ?",
        "input": "L'assurance auto couvre le conducteur et le véhicule en cas d'accident, de vol ou de dommages matériels.",
        "output": "L'assurance auto couvre le conducteur et le véhicule en cas d'accident, de vol ou de dommages matériels.",
    },
    {
        "instruction": "Quelles sont les garanties principales d'une assurance auto ?",
        "input": "Les garanties principales incluent : responsabilité civile obligatoire, dommages tous accidents, bris de glace, assistance 0 km et protection juridique.",
        "output": "Les garanties principales sont la responsabilité civile (obligatoire), les dommages tous accidents, le bris de glace, l'assistance 0 km et la protection juridique.",
    },
    {
        "instruction": "Quelle est la différence entre la responsabilité civile et la garantie tous risques ?",
        "input": "La responsabilité civile couvre les dommages causés à un tiers mais ne couvre pas les dommages du véhicule assuré. La garantie tous risques prend en charge les dommages même en cas de responsabilité du conducteur.",
        "output": "La responsabilité civile couvre uniquement les dommages causés à un tiers, pas ceux de votre propre véhicule. La garantie tous risques prend en charge les dommages de votre véhicule, même si vous êtes responsable de l'accident.",
    },
    {
        "instruction": "Combien de temps ai-je pour déclarer un accident de voiture ?",
        "input": "En cas de sinistre, l'assuré doit déclarer l'accident sous 5 jours ouvrés et transmettre un constat amiable.",
        "output": "Vous devez déclarer l'accident sous 5 jours ouvrés et transmettre un constat amiable à votre assureur.",
    },
    {
        "instruction": "Comment fonctionne le bonus-malus ?",
        "input": "Le bonus-malus influence le montant de la prime d'assurance. Un bonus maximal de 0,5 peut être atteint après 13 ans sans accident responsable.",
        "output": "Le bonus-malus influence le montant de votre prime : plus vous accumulez d'années sans accident responsable, plus votre bonus augmente, jusqu'à un maximum de 0,5 après 13 ans.",
    },

    # --- assurance_habitation.txt ---
    {
        "instruction": "Qu'est-ce que couvre une assurance habitation ?",
        "input": "L'assurance habitation protège le logement et les biens contre les dégâts des eaux, incendies, vols, explosions et tempêtes.",
        "output": "Elle protège votre logement et vos biens contre les dégâts des eaux, les incendies, les vols, les explosions et les tempêtes.",
    },
    {
        "instruction": "L'assurance habitation couvre-t-elle les dommages causés à mes voisins ?",
        "input": "Elle inclut également une responsabilité civile couvrant les dommages causés aux voisins ou à la copropriété.",
        "output": "Oui, elle inclut une responsabilité civile qui couvre les dommages que vous pourriez causer à vos voisins ou à la copropriété.",
    },
    {
        "instruction": "Qu'est-ce que la garantie MRH inclut ?",
        "input": "Le contrat Multirisques Habitation (MRH) comprend les garanties : incendie, dégâts des eaux, vol, vandalisme, bris de glace, catastrophes naturelles et attentats.",
        "output": "Le contrat Multirisques Habitation (MRH) comprend les garanties incendie, dégâts des eaux, vol, vandalisme, bris de glace, catastrophes naturelles et attentats.",
    },
    {
        "instruction": "Que dois-je faire en cas de dégât des eaux chez moi ?",
        "input": "En cas de dégât des eaux, l'assuré doit prévenir son assureur sous 5 jours et fournir photos, facture et descriptif des dommages.",
        "output": "Vous devez prévenir votre assureur sous 5 jours, et fournir des photos, une facture ainsi qu'un descriptif des dommages.",
    },
    {
        "instruction": "Puis-je assurer ma piscine ou mes équipements électriques ?",
        "input": "Certaines options permettent d'assurer les équipements électriques, les objets de valeur ou les piscines.",
        "output": "Oui, certaines options permettent d'assurer les équipements électriques, les objets de valeur ou les piscines.",
    },

    # --- assurance_sante.txt ---
    {
        "instruction": "Qu'est-ce qu'une assurance santé couvre ?",
        "input": "L'assurance santé couvre partiellement ou totalement les frais médicaux selon les garanties souscrites.",
        "output": "Elle couvre partiellement ou totalement vos frais médicaux, selon les garanties que vous avez souscrites.",
    },
    {
        "instruction": "Quels frais médicaux sont remboursés par une mutuelle ?",
        "input": "Les principaux postes remboursés sont : consultations, pharmacie, hospitalisation, optique et dentaire.",
        "output": "Les principaux postes remboursés sont les consultations, la pharmacie, l'hospitalisation, l'optique et le dentaire.",
    },
    {
        "instruction": "Quel est le rôle d'une mutuelle santé par rapport à la Sécurité sociale ?",
        "input": "Les mutuelles santé complètent la Sécurité sociale et permettent une meilleure prise en charge.",
        "output": "Une mutuelle santé complète les remboursements de la Sécurité sociale, pour une meilleure prise en charge de vos frais médicaux.",
    },
    {
        "instruction": "Qu'est-ce que le tiers payant ?",
        "input": "Certaines formules incluent le tiers payant, évitant d'avancer les frais chez les professionnels de santé.",
        "output": "Le tiers payant vous évite d'avancer les frais chez les professionnels de santé -- certaines formules l'incluent.",
    },
    {
        "instruction": "Comment obtenir un remboursement de soins de santé ?",
        "input": "Pour un remboursement, l'assuré doit transmettre la feuille de soins ou utiliser la télétransmission automatique.",
        "output": "Vous devez transmettre votre feuille de soins, ou passer par la télétransmission automatique si elle est disponible.",
    },

    # --- conditions_generales_auto.txt ---
    {
        "instruction": "Dans quels cas l'assureur peut-il résilier mon contrat auto ?",
        "input": "L'assureur peut résilier le contrat en cas de fausse déclaration ou de non-paiement.",
        "output": "L'assureur peut résilier votre contrat en cas de fausse déclaration ou de non-paiement de vos cotisations.",
    },
    {
        "instruction": "Quelles sont mes obligations envers mon véhicule assuré ?",
        "input": "L'assuré doit maintenir le véhicule en bon état de fonctionnement et signaler tout changement de situation.",
        "output": "Vous devez maintenir votre véhicule en bon état de fonctionnement, et signaler tout changement de situation à votre assureur.",
    },
    {
        "instruction": "Quelles sont les principales exclusions d'une assurance auto ?",
        "input": "Les exclusions principales sont : conduite en état d'ivresse, absence de permis valide, participation à des compétitions non autorisées.",
        "output": "Les exclusions principales sont la conduite en état d'ivresse, l'absence de permis valide, et la participation à des compétitions non autorisées.",
    },
    {
        "instruction": "Mon assurance auto me couvre-t-elle à l'étranger ?",
        "input": "Les garanties sont applicables uniquement sur le territoire couvert par le contrat.",
        "output": "Pas nécessairement -- les garanties ne s'appliquent que sur le territoire couvert par votre contrat, il faut vérifier les conditions pour une couverture à l'étranger.",
    },

    # --- contrats_definitions.txt (glossaire) ---
    {
        "instruction": "Qu'est-ce qu'une prime d'assurance ?",
        "input": "Prime d'assurance : montant payé périodiquement par l'assuré.",
        "output": "La prime d'assurance est le montant que vous payez périodiquement (mensuellement ou annuellement) à votre assureur.",
    },
    {
        "instruction": "Que signifie le mot garantie dans un contrat d'assurance ?",
        "input": "Garantie : engagement de l'assureur de couvrir un risque défini.",
        "output": "Une garantie est l'engagement pris par l'assureur de couvrir un risque défini dans votre contrat.",
    },
    {
        "instruction": "Qu'est-ce qu'une franchise en assurance ?",
        "input": "Franchise : somme restant à la charge de l'assuré en cas de sinistre.",
        "output": "La franchise est la somme qui reste à votre charge en cas de sinistre, le reste étant remboursé par l'assureur.",
    },
    {
        "instruction": "Qu'est-ce qu'un sinistre ?",
        "input": "Sinistre : événement imprévu générant une demande d'indemnisation.",
        "output": "Un sinistre est un événement imprévu (accident, vol, incendie...) qui donne lieu à une demande d'indemnisation auprès de l'assureur.",
    },
    {
        "instruction": "Qu'est-ce qu'une indemnisation ?",
        "input": "Indemnisation : montant versé par l'assureur suite à un sinistre.",
        "output": "L'indemnisation est le montant versé par l'assureur pour compenser les dommages subis suite à un sinistre.",
    },
    {
        "instruction": "Qu'est-ce qu'une exclusion dans un contrat d'assurance ?",
        "input": "Exclusion : situation non couverte par le contrat.",
        "output": "Une exclusion est une situation ou un événement qui n'est pas couvert par votre contrat d'assurance.",
    },

    # --- reclamations_client.txt (le modèle apprend à réagir à une vraie plainte) ---
    {
        "instruction": "Je ne suis pas satisfait du traitement de mon dossier de sinistre. Le montant proposé ne correspond pas aux dommages réels.",
        "input": "Le service de gestion peut demander un expert pour évaluer les dommages.",
        "output": "Je comprends votre frustration. Vous pouvez demander une contre-expertise : le service de gestion peut faire intervenir un expert pour réévaluer les dommages et le montant proposé.",
    },
    {
        "instruction": "J'ai été prélevé deux fois ce mois-ci pour mon contrat habitation.",
        "input": "",
        "output": "Cela ressemble à une erreur de prélèvement. Contactez le service client pour signaler le double prélèvement et demander le remboursement du trop-perçu.",
    },
    {
        "instruction": "Je demande une révision de mon tarif auto, car mon bonus n'a pas été pris en compte.",
        "input": "Le bonus-malus influence le montant de la prime d'assurance. Un bonus maximal de 0,5 peut être atteint après 13 ans sans accident responsable.",
        "output": "Votre bonus-malus influence directement votre prime. Si le vôtre n'a pas été pris en compte, demandez une révision de votre tarif en fournissant votre relevé d'information à votre assureur.",
    },
    {
        "instruction": "Impossible de contacter le service client, personne ne répond depuis trois jours.",
        "input": "",
        "output": "Je suis désolé pour cette difficulté. Essayez de passer par votre espace client en ligne pour laisser une demande écrite, un conseiller devrait vous répondre plus rapidement par ce biais.",
    },
    {
        "instruction": "Mon remboursement d'optique n'a toujours pas été effectué malgré l'envoi des justificatifs.",
        "input": "Pour un remboursement, l'assuré doit transmettre la feuille de soins ou utiliser la télétransmission automatique.",
        "output": "Vérifiez que votre feuille de soins a bien été transmise ou que la télétransmission a fonctionné. Si c'est le cas, contactez le service client pour relancer le traitement de votre remboursement.",
    },

    # --- sinistres_procedure.txt ---
    {
        "instruction": "Comment déclarer un sinistre auto ?",
        "input": "Pour déclarer un sinistre auto, l'assuré doit remplir un constat amiable et le transmettre sous 5 jours.",
        "output": "Pour déclarer un sinistre auto, remplissez un constat amiable et transmettez-le à votre assureur sous 5 jours.",
    },
    {
        "instruction": "Que faire en cas de vol de mon véhicule ?",
        "input": "En cas de vol, une déclaration préalable doit être faite au commissariat dans les 24 heures.",
        "output": "En cas de vol, vous devez d'abord faire une déclaration au commissariat dans les 24 heures, avant de déclarer le sinistre à votre assureur.",
    },
    {
        "instruction": "Quels documents dois-je fournir pour un sinistre habitation ?",
        "input": "Pour un sinistre habitation, des photos et factures doivent être jointes à la déclaration.",
        "output": "Vous devez joindre des photos et des factures à votre déclaration de sinistre habitation.",
    },
    {
        "instruction": "Un expert va-t-il évaluer mes dommages après un sinistre ?",
        "input": "Le service de gestion peut demander un expert pour évaluer les dommages.",
        "output": "Oui, le service de gestion peut demander l'intervention d'un expert pour évaluer précisément vos dommages.",
    },
    {
        "instruction": "Combien de temps faut-il pour être indemnisé après un sinistre ?",
        "input": "Le délai d'indemnisation varie selon la nature du sinistre, de 10 jours pour un bris de glace à 30 jours pour un incendie.",
        "output": "Le délai d'indemnisation dépend du type de sinistre : environ 10 jours pour un bris de glace, jusqu'à 30 jours pour un incendie.",
    },

    # --- service_client_faq.txt ---
    {
        "instruction": "Comment obtenir une attestation d'assurance ?",
        "input": "Vous pouvez télécharger votre attestation depuis votre espace client ou en faire la demande par téléphone.",
        "output": "Vous pouvez télécharger votre attestation depuis votre espace client, ou en faire la demande par téléphone.",
    },
    {
        "instruction": "Quels moyens de paiement sont acceptés ?",
        "input": "Carte bancaire, prélèvement automatique, virement ou chèque selon les contrats.",
        "output": "Selon votre contrat, vous pouvez payer par carte bancaire, prélèvement automatique, virement ou chèque.",
    },
    {
        "instruction": "Puis-je résilier mon contrat auto à tout moment ?",
        "input": "Oui, grâce à la loi Hamon, vous pouvez résilier après un an de contrat, sans frais.",
        "output": "Oui, grâce à la loi Hamon, vous pouvez résilier votre contrat après un an, sans frais.",
    },
    {
        "instruction": "Comment modifier mes informations personnelles ?",
        "input": "Depuis votre espace client, rubrique « Profil », vous pouvez changer votre adresse, votre RIB ou vos coordonnées.",
        "output": "Depuis votre espace client, dans la rubrique « Profil », vous pouvez modifier votre adresse, votre RIB ou vos coordonnées.",
    },

    # --- synthèse (fixe directement le cas d'hallucination observé) ---
    {
        "instruction": "C'est quoi une assurance ?",
        "input": "Une garantie est l'engagement de l'assureur de couvrir un risque défini. Prime d'assurance : montant payé périodiquement par l'assuré. Sinistre : événement imprévu générant une demande d'indemnisation.",
        "output": "Une assurance est un contrat par lequel un assureur s'engage, en échange d'une prime payée régulièrement, à couvrir un risque défini (accident, vol, incendie, maladie...) et à vous indemniser en cas de sinistre. Il existe différents types d'assurance -- auto, habitation, santé -- chacune couvrant des risques spécifiques.",
    },
    {
        "instruction": "Qu'est-ce qu'une assurance, en résumé ?",
        "input": "",
        "output": "C'est un contrat de protection : vous payez une prime, et en échange l'assureur s'engage à vous indemniser si un événement couvert (sinistre) se produit.",
    },

    # --- exemples de refus poli (hors-sujet, sans contexte) ---
    {
        "instruction": "Quel est le prix d'une Tesla Model 3 ?",
        "input": "",
        "output": "Je suis un assistant spécialisé dans les questions d'assurance et je n'ai pas cette information. Je peux t'aider sur tes garanties, sinistres, remboursements ou résiliation.",
    },
    {
        "instruction": "Quelle est la capitale de l'Australie ?",
        "input": "",
        "output": "Ce n'est pas mon domaine -- je suis un assistant spécialisé dans l'assurance. Pose-moi plutôt une question sur tes contrats, tes garanties ou tes sinistres.",
    },
]


def main():
    builder = DatasetBuilder()
    builder.save_examples(EXAMPLES, OUTPUT_PATH)
    print(f"{len(EXAMPLES)} exemples Q/R écrits dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

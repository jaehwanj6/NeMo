# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from omegaconf import OmegaConf
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.core.config import hydra_runner
# from nemo.collections.nlp.data.language_modeling.megatron.retrieval_service import FaissRetrievalService


try:
    from apex.transformer import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


@hydra_runner(config_path="conf", config_name="megatron_retro_mutransfer")
def main(cfg) -> None:

    # trainer required for restoring model parallel models
    trainer = Trainer(plugins=NLPDDPPlugin(), **cfg.trainer)

    for k in [1,2,3,4,5,10,20,30,40,50]: 
        model = MegatronRetrievalModel.restore_from(restore_path=cfg.restore_from_path, trainer=trainer)
        OmegaConf.set_struct(model.cfg, False)
        model.cfg.data.neighbors = k
        model.cfg.data.data_prefix = ['/shared-volume/retro_wei_wiki_text_document']
        # model.cfg.data.knn_index = [' '.join(['/shared-volume/knn_map_wei_wiki_50_start_{}.idx'.format(el) for el in range(0, 1321, 60)])]
        # model.cfg.data.knn_index = ['/shared-volume/knn_map_wei_wiki.idx']
        model.cfg.data.knn_index = ['/shared-volume/knn_final.save']
        model.cfg.data.retrieval_prefix = '/shared-volume/retro_wei_wiki_text_document'
        # model.cfg.data.knn_map_size = -1
        # model.cfg.data.knn_map_size = 3072000
        OmegaConf.set_struct(model.cfg, True)
        trainer.test(model)
        


    # model.cfg.data.neighbors = 1

    # model.freeze()
    
    # retriever = FaissRetrievalService(
    #                                   model.tokenizer, 
    #                                   cfg.faiss_index_path, 
    #                                   cfg.model.data.retrieval_prefix, 
    #                                   cfg.model.data.data_impl, 
    #                                   cfg.model.data.skip_warmup, 
    #                                   cfg.model.chunk_size, 
    #                                   cfg.model.data.neighbors, 
    #                                   devices='0,1'
    #                                 )

    # model.retriever = retriever
    

    # # response = model.generate(OmegaConf.to_container(cfg.prompts), length_params, sampling_params)
    # prompt = ' obstruction to keep the bill from moving to a vote. The senators had also made amendments to the bill to exclude both Marineland of Canada and the Vancouver Aquarium from being covered by the bill. After three years, the eventual outcome was not yet known in October 2018.\n\nIn popular culture\nThe Vancouver Aquarium was featured frequently in the 1980s Canadian series, Danger Bay, which followed the day to day exploits of the Roberts family, led by Grant "Doc" Roberts, a marine veterinarian and his two children, Nicole and Jonah.\n\nA YouTube video featuring two sea otters "holding hands" was recorded at the Vancouver Aquarium. The two sea otters are Nyac and Milo. Nyac died on September 23, 2008. She was one of the last surviving sea otters of the 1989 Exxon Valdez oil spill. The video has been viewed over 19 million times on YouTube. As a result, the Vancouver Aquarium created a live sea otter cam on their website. The YouTube video was originally recorded by Cynthia Holmes. Milo died on January 12, 2012.\n\nThe Vancouver Aquarium was also featured in the family film Andre (1994), and romantic comedy Good Luck Chuck (2007), as Cam\'s workplace. Television movie The Suite Life Movie (2011) used the aquarium as the research firm where Cody Martin interns.\n\nOn September 5, 2008, Hayden Panettiere appeared on the Late Show with David Letterman and talked about her visit with the rescue dolphins at the Vancouver Aquarium.\n\nThe song "Baby Beluga" by Raffi was inspired by Kavna, a beluga that he saw while visiting the Vancouver Aquarium.\n\nReferences\n\nBibliography\n\n This is a history of the aquarium as told by the founding and current presidents of the aquarium.\n\nWaters is a magazine published by Canada Wide Media Limited for the official members of the Vancouver Aquarium. It is published three times a year.\n\nExternal links\n\nCategory:Stanley Park\nCategory:Buildings and structures in Vancouver\nCategory:Aquaria in Canada\nCategory:Tourist attractions in Vancouver\nCategory:Wildlife rehabilitation\nCategory:Marine mammal rehabilitation and conservation centersOrzeszkowo, West Pomeranian Voivodeship\n\nOrzeszkowo  () is a village in the administrative district of Gmina Resko, within Łobez County, West Pomeranian Voivodeship, in north-western Poland. It lies approximately  north-east of Resko,  north of Łobez, and  north-east of the regional capital Szczecin.\n\nBefore 1945 the area was part of Germany. For the history of the region, see History of Pomerania.\n\nThe village has a population of 30.\n\nReferences\n\nOrzeszkowoSouleymane Youla\n\nSouleymane Youla (born 29 November 1981) is a Guinean football player, who plays for Ronse. He has Turkish citizenship with the name Süleyman Yula.\n\nCareer\n\nClub\nYoula\'s professional career started in Belgium in 1999, when Lokeren signed him to replace the departed Jan Koller, who had moved to Anderlecht. Scoring 9 goals in 14 matches, Youla played a successful season, before also being signed by Anderlecht. At Anderlecht, he faced fierce competition in the likes of Koller, Tomasz Radzinski, Aruna Dindane and Oleg Iachtchouk. He only remained for one season but is remembered by Anderlecht fans for his injury time winner against PSV Eindhoven in the 2000–01 UEFA Champions League, allowing Anderlecht to qualify for the Round of 16 as group winners. Youla moved to Turkey where he signed for Gençlerbirliği SK and played five seasons as a first team player. He signed for Beşiktaş J.K., got much less playing time and was loaned out for one season to French side FC Metz. At Metz, he was noticed by Lille, who signed him in 2007. After again a season where he did not receive much opportunities, he was loaned out back to Turkish side Eskişehirspor where he partnered up with Ümit Karan. The following season, the transfer was made permanent. Thereafter, Youla enjoyed two more seasons in Turkey, with Denizlispor and Orduspor. Following the 2010–11 season, he stayed unemployed until November 2012, when he was hired by Belgian team Sint-Niklaas to help the team remain in the Belgian Second Division. On 11 June 2016, Youla signed for North American Soccer League side Indy Eleven.\n\nInternational\nHe was part of the Guinean 2004 African Nations Cup team who finished second in their group in the first round of competition. The team progressed to the quarter finals, where they lost to Mali.\n\nCareer statistics\n\nInternational\n\nStatistics accurate as of match played 14 November 2009\n\nInternational goals\n\nReferences\n\nExternal links\n \n \n\nCategory:1981 births\nCategory:Living people\nCategory:Guinean footballers\nCategory:Guinean expatriate footballers\nCategory:Association football forwards\nCategory:Lille OSC players\nCategory:FC Metz players\nCategory:K.S.C. Lokeren Oost-Vlaanderen players\nCategory:R.S.C. Anderlecht players\nCategory:Beşiktaş J.K. footballers\nCategory:Gençlerbirliği S.K. footballers\nCategory:Eskişehirspor footballers\nCategory:Denizlispor footballers\nCategory:Orduspor footballers\nCategory:Amiens SC players\nCategory:Expatriate footballers in Belgium\nCategory:Expatriate footballers in Turkey\nCategory:Expatriate footballers in France\nCategory:Guinea international footballers\nCategory:2004 African Cup of Nations players\nCategory:2008 Africa Cup of Nations players\nCategory:Ligue 1 players\nCategory:Belgian First Division A players\nCategory:Süper Lig players\nCategory:Championnat National players\nCategory:Sportspeople from Conakry\nCategory:Guinean expatriate sportspeople in Ivory Coast\nCategory:Stade d\'Abidjan players\nCategory:Expatriate footballers in Ivory Coast\nCategory:Guinean expatriate sportspeople in Turkey\nCategory:Turkish people of Guinean descent\nCategory:Naturalized citizens of Turkey\nCategory:Expatriate footballers in Hungary\nCategory:Guinean expatriate sportspeople in Hungary\nCategory:Budapest Honvéd FC players\nCategory:Nemzeti Bajnokság I players\nCategory:Indy Eleven players\nCategory:North American Soccer League players\nCategory:Sportkring Sint-Niklaas playersRhodophthitus\n\nRhodophthitus is a genus of moths in the family Geometridae described by Arthur Gardiner Butler in 1880.\n\nSpecies\nSome species of this genus are:\nRhodophthitus anamesa (Prout, 1915)\nRhodophthitus arichanaria D. S. Fletcher, 1978\nRhodophthitus atacta Prout, 1922\nRhodophthitus atricoloraria (Mabille, 1890)\nRhodophthitus barlowi (Prout, 1922)\nRhodophthitus betsileanus Herbulot, 1965\nRhodophthitus castus Warren, 1904\nRhodophthitus commaculata (Warren, 1897)\nRhodophthitus formosus Butler, 1880\nRhodophthitus myriostictus Prout, 1915\nRhodophthitus procellosa Warren, 1905\nRhodophthitus pseudabraxas Carcasson, 1964\nRhodophthitus roseovittata (Butler, 1895)\nRhodophthitus rudicornis (Butler, 1898)\nRhodophthitus simplex Warren, 1897\nRhodophthitus thapsinus Prout, 1931\nRhodophthitus tricoloraria (Mabille, 1890)\nRhodophthitus unca (Le Cerf, 1922)\n\nReferences\n\nCategory:GeometridaeBlack and White (EP)\n\nBlack and White is the first EP by Italian melodic metal band Dimmi Argus. It was released on 6 April 2010 and produced by Dimitar Argirov.'

    # response = model.generate([prompt], length_params, sampling_params)
    # for sent in response['sentences']: 
    #     print(sent)
    #     print('\n' + '-' * 100 + '\n')

    


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

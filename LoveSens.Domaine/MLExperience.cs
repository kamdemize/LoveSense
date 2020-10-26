using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using System;

namespace LoveSense.Domaine
{
    public class MLExperience: ResponseBase
    {
        [JsonProperty("date_experience")]
        public DateTime DateExperience { get; set; }

        [JsonConverter(typeof(StringEnumConverter))]
        public ExperienceType ExperienceType { get; set; }

        [JsonProperty("code_experience")]
        public string Code { get; set; }

        [JsonProperty("score")]
        public decimal Score { get; set; }

        [JsonProperty("time_training")]
        public decimal TrainingTime { get; set; }

        [JsonProperty("time_test")]
        public decimal TestTime { get; set; }

        [JsonProperty("erreur_experience")]
        public string Error { get; set; }
    }
}

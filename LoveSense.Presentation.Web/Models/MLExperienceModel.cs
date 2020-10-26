using LoveSense.Domaine;
using System;

namespace LoveSense.Presentation.Web.Models
{
    public class MLExperienceModel
    {
        public DateTime DateExperience { get; set; }
        public string ExperienceType { get; set; }
        public string Code { get; set; }
        public decimal Score { get; set; }
        public decimal TrainingTime { get; set; }
        public decimal TestTime { get; set; }
        public string Error { get; set; }
    }
}

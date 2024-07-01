using UnityEngine;

namespace Env5
{
    public class MORLEnvSimpleController : MonoBehaviour, EnvController
    {
        public Transform agent;
        public Transform goal;
      
        private float groundY = 0f;
        private float width = 40f;

        float part1 = 0.25f;
        float part2 = 0.25f;
        float part3 = 0.25f;
        float part4 = 0.25f;
        float playerScale = 1f;
        float buttonHeight = 0.0002f;
        float buttonScale = 4f;
        float margin = 1f;

        float x0;
        float x1;
        float x2;
        float x3;
        float x4;
        float bridgeZ;

        public float Width => width;
        
        void Awake()
        {
            Initialize();
        }

        public bool AtGoal()
        {
            return agent.gameObject.GetComponent<CollisionDetector>().Touching(goal.gameObject);
        }

        public void Initialize()
        {
            x0 = -Width / 2;
            x1 = x0 + Width * part1;
            x2 = x1 + Width * part2;
            x3 = x2 + Width * part3;
            x4 = x3 + Width * part4;

            float minX = x0 + margin + playerScale / 2;
            float maxX = x4 - margin - playerScale / 2;
            float z0 = -Width / 2;
            float z1 = Width / 2;
            float minZ = z0 + margin + playerScale / 2;
            float maxZ = z1 - margin - playerScale / 2;

            agent.gameObject.GetComponent<Rigidbody>().velocity = Vector3.zero;
            goal.localPosition = new Vector3(Random.Range(x3 + buttonScale / 2, x4 - buttonScale / 2), 0.001f, Random.Range(z0 + buttonScale / 2, z1 - buttonScale / 2));
            agent.localPosition = new Vector3(Random.Range(minX, maxX), 0.06f, Random.Range(minZ, maxZ));
        }

        public void Reset()
        {
            Initialize();
        }
    }
}